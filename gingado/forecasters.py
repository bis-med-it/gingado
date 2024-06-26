from __future__ import annotations  # Allows forward annotations in Python < 3.10

import keras

__all__ = ['MultiInputContEmbedding', 'MultiInputCategEmbedding', 'InputTFT', 'GatedLinearUnit', 'GatedResidualNetwork', 'StaticVariableSelection', 'TemporalVariableSelection', 'TemporalFeatures', 'ScaledDotProductAttention', 'InterpretableMultiHeadAttention', 'TFT']

class MultiInputContEmbedding(keras.Layer):
    def __init__(
        self, 
        d_model:int, # Embedding size, $d_\text{model}$
        **kwargs
    ):
        "Embeds multiple continuous variables, each with own embedding space"
        
        super(MultiInputContEmbedding, self).__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        super(MultiInputContEmbedding, self).build(input_shape)

        # input_shape: (batch_size, num_time_steps, num_variables)
        num_variables = input_shape[-1]
        
        self.kernel = self.add_weight(
            shape=(num_variables, self.d_model),
            initializer='uniform',
            name='kernel'
        )

        self.bias = self.add_weight(
            shape=(self.d_model,),
            initializer='zeros',
            name='bias'
        )
    
    def call(
        self,
        inputs # Data of shape: (batch size, num time steps, num variables)
    ):     
        "Output shape: (batch size, num time steps, num variables, d_model)"

        # Applying the linear transformation to each time step
        output = keras.ops.einsum('bti,ij->btij', inputs, self.kernel)
        output += self.bias
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], self.d_model

class MultiInputCategEmbedding(keras.Layer):
    def __init__(
        self,
        d_model:int, # Embedding size, $d_\text{model}$
        cardinalities:dict, # Variable: cardinality in training data
        **kwargs
    ):
        "Embeds multiple categorical variables, each with own embedding function"
        super(MultiInputCategEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.cardinalities = cardinalities
        
    def build(self, input_shape):
        super(MultiInputCategEmbedding, self).build(input_shape)
        if len(self.cardinalities.keys()) != input_shape[-1]:
            raise ValueError("`cardinalities` should have as many elements as the input data's variables.")
        
        self.embed_layer = {var:
            keras.Sequential([
                keras.layers.Embedding(
                    input_dim=cardin,
                    output_dim=self.d_model,
                    mask_zero=True,
                    name="input_embed_" + var.replace(" ", "_")
                )
            ]) for var, cardin in self.cardinalities.items()
        }
        super(MultiInputCategEmbedding, self).build(input_shape)

    def call(
        self,
        inputs # Data of shape: (batch size, num time steps, num variables)
    ):
        "Output shape: (batch size, num time steps, num variables, d_model)"
        embeds = [
            self.embed_layer[var](inputs[:,:,idx])
            for idx, var in enumerate(self.cardinalities.keys())
        ]
        return keras.ops.stack(embeds, axis=2) # keras.ops.concatenate(embeds, axis=-1)

class InputTFT(keras.Layer):
    def __init__(
        self,
        d_model:int=16, # Embedding size, $d_\text{model}$
        **kwargs
    ):
        "Input layer for the Temporal Fusion Transformer model"
        super(InputTFT, self).__init__(**kwargs)
        self.d_model = d_model
        
        self.flat = keras.layers.Flatten()
        self.concat = keras.layers.Concatenate(axis=2)
    
    def build(self, input_shape):
        self.cont_hist_embed = MultiInputContEmbedding(
            self.d_model,
            name="embed_continuous_historical_vars"
        )
        self.cat_hist_embed = MultiInputCategEmbedding(
            self.d_model, 
            cardinalities=cardin_hist, # TODO: incorporate the calculation of the cardinality
            name="embed_categ_historical_vars"
        )
        self.cat_fut_embed = MultiInputCategEmbedding(
            self.d_model, 
            # Note below the same categorical variables are just the months in an year. 
            # This situation may not apply to all cases.
            # More complex models using other categorical future known data might require
            # another cardinalities dictionary.
            cardinalities={'Month of Year': 13},
            name="embed_categ_knownfuture_vars"
        )
        self.cat_stat_embed = MultiInputCategEmbedding(
            self.d_model, 
            cardinalities=cardin_stat, # TODO: incorporate the calculation of the cardinality
            name="embed_categ_static_vars"
        )
        super(InputTFT, self).build(input_shape)

    def call(
        self, 
        # List of data with shape: [(batch size, num hist time steps, num continuous hist variables), (batch size, num hist time steps, num categorical hist variables), (batch size, num static variables), (batch size, num future time steps, num categorical future variables)]
        input:list 
    ):
        """List of output with shape: [
            (batch size, num hist time steps, num historical variables, d_model),
            (batch size, num future time steps, num future variables, d_model)
            (batch size, one, num static variables, d_model),
        ]"""
        cont_hist, cat_hist, cat_fut, cat_stat = input
        if len(cat_stat.shape) == 2:
            cat_stat = keras.ops.expand_dims(cat_stat, axis=-1)

        cont_hist = self.cont_hist_embed(cont_hist)
        #cont_hist = keras.ops.swapaxes(cont_hist, axis1=2, axis2=3)

        cat_hist = self.cat_hist_embed(cat_hist)
        #cat_hist = self.flat(cat_hist)
            
        cat_fut = self.cat_fut_embed(cat_fut)
        #cat_fut = self.flat(cat_fut)
        
        cat_stat = self.cat_stat_embed(cat_stat)
        #cat_stat = self.flat(cat_stat)

        # (batch size / (num time steps * (num historical + future variables) + num static variables) * embedding size)
        hist = self.concat([cont_hist, cat_hist])
        
        return hist, cat_fut, cat_stat

class GatedLinearUnit(keras.Layer):
    def __init__(
        self,
        d_model:int=16, # Embedding size, $d_\text{model}$
        dropout_rate:float|None=None, # Dropout rate
        use_time_distributed:bool=True, # Apply the GLU across all time steps?
        activation:str|callable=None, # Activation function
        **kwargs
    ):
        "Gated Linear Unit dynamically gates input data"
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        self.activation = activation

    def build(self, input_shape):
        super(GatedLinearUnit, self).build(input_shape)
        self.dropout = keras.layers.Dropout(self.dropout_rate) if self.dropout_rate is not None else None
        self.activation_layer = keras.layers.Dense(self.d_model, activation=self.activation)
        self.gate_layer = keras.layers.Dense(self.d_model, activation='sigmoid')
        self.multiply = keras.layers.Multiply()

        if self.use_time_distributed:
            self.activation_layer = keras.layers.TimeDistributed(self.activation_layer)
            self.gate_layer = keras.layers.TimeDistributed(self.gate_layer)

    def call(
        self, 
        inputs, 
        training=None
    ):
        """List of outputs with shape: [
            (batch size, ..., d_model),
            (batch size, ..., d_model)
        ]"""
        if self.dropout is not None and training:
            inputs = self.dropout(inputs)

        activation_output = self.activation_layer(inputs)
        gate_output = self.gate_layer(inputs)
        return self.multiply([activation_output, gate_output]), gate_output

    def get_config(self):
        config = super(GatedLinearUnit, self).get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': self.use_time_distributed,
            'activation': self.activation
        })
        return config

class GatedResidualNetwork(keras.layers.Layer):
    def __init__(
        self, 
        d_model:int=16, # Embedding size, $d_\text{model}$
        output_size=None, 
        dropout_rate=None, 
        use_time_distributed=True, 
        **kwargs
    ):
        "Gated residual network"
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.d_model = d_model
        self.output_size = output_size if output_size is not None else d_model
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed

    def build(self, input_shape):
        super(GatedResidualNetwork, self).build(input_shape)
        self.dense = keras.layers.Dense(self.output_size)
        self.hidden_dense = keras.layers.Dense(self.d_model)
        self.hidden_activation = keras.layers.Activation('elu')
        self.context_dense = keras.layers.Dense(self.d_model, use_bias=False)
        self.gating_layer = GatedLinearUnit(
            d_model=self.output_size, 
            dropout_rate=self.dropout_rate, 
            use_time_distributed=self.use_time_distributed, 
            activation=None)
        self.add = keras.layers.Add()
        self.l_norm = keras.layers.LayerNormalization()

        if self.use_time_distributed:
            self.dense = keras.layers.TimeDistributed(self.dense)
            self.hidden_dense = keras.layers.TimeDistributed(self.hidden_dense)
            self.context_dense = keras.layers.TimeDistributed(self.context_dense)

    def call(self, inputs, additional_context=None, training=None):
        # Setup skip connection
        skip = self.dense(inputs) if self.output_size != self.d_model else inputs
        
        # 1st step: eta2
        hidden = self.hidden_dense(inputs)

        # Context handling
        if additional_context is not None:
            hidden += self.context_dense(additional_context)

        hidden = self.hidden_activation(hidden)

        # 2nd step: eta1 and 3rd step
        gating_layer, gate = self.gating_layer(hidden)
        
        # Final step
        GRN = self.add([skip, gating_layer])
        GRN = self.l_norm(GRN)

        return GRN, gate

    def get_config(self):
        config = super(GatedResidualNetwork, self).get_config()
        config.update({
            'd_model': self.d_model,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': self.use_time_distributed
        })
        return config

class StaticVariableSelection(keras.Layer):
    def __init__(
        self, 
        d_model:int=16, # Embedding size, $d_\text{model}$
        dropout_rate:float=0., 
        **kwargs
    ):
        "Static variable selection network"
        super(StaticVariableSelection, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # Define GRNs for the transformed embeddings
        self.grns_transformed_embeddings = []  # This will be a list of GRN layers

        self.flat = keras.layers.Flatten()
        self.softmax = keras.layers.Activation('softmax')
        self.mult = keras.layers.Multiply()

    def build(self, input_shape):
        super(StaticVariableSelection, self).build(input_shape)
        
        num_static = input_shape[2]

        # Define the GRN for the sparse weights
        self.grn_sparse_weights = GatedResidualNetwork(
            d_model=self.d_model,
            output_size=num_static,
            use_time_distributed=False
        )

        for i in range(num_static):
            # Create a GRN for each static variable
            self.grns_transformed_embeddings.append(
                GatedResidualNetwork(self.d_model, use_time_distributed=False)
            )

    def call(self, inputs, training=None):
        _, _, num_static, _ = inputs.shape # batch size / one time step (since it's static) / num static variables / d_model

        flattened = self.flat(inputs)

        # Compute sparse weights
        grn_outputs, _ = self.grn_sparse_weights(flattened, training=training)
        sparse_weights = self.softmax(grn_outputs)
        sparse_weights = keras.ops.expand_dims(sparse_weights, axis=-1)

        # Compute transformed embeddings
        transformed_embeddings = []
        for i in range(num_static):
            embed, _ = self.grns_transformed_embeddings[i](inputs[:, 0, i:i+1, :], training=training)
            transformed_embeddings.append(embed)
        transformed_embedding = keras.ops.concatenate(transformed_embeddings, axis=1)

        # Combine with sparse weights
        combined = self.mult([sparse_weights, transformed_embedding])
        static_vec = keras.ops.sum(combined, axis=1)

        return static_vec, sparse_weights

    def get_config(self):
        config = super(StaticVariableSelection, self).get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate
        })
        return config

class TemporalVariableSelection(keras.Layer):
    def __init__(
        self, 
        d_model:int=16, # Embedding size, $d_\text{model}$
        dropout_rate:float=0., 
        **kwargs
    ):
        "Temporal variable selection"
        super(TemporalVariableSelection, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        self.mult = keras.layers.Multiply()

    def build(self, input_shape):
        super(TemporalVariableSelection, self).build(input_shape)
        self.batch_size, self.time_steps, self.num_input_vars, self.d_model = input_shape[0]

        self.var_sel_weights = GatedResidualNetwork(
            d_model=self.d_model,
            output_size=self.num_input_vars,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
        )
        self.softmax = keras.layers.Activation('softmax')
    
        # Create a GRN for each temporal variable
        self.grns_transformed_embeddings = GatedResidualNetwork(
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True
        )

    def call(
        self, 
        inputs, # List of temporal embeddings, static context
        training=None
    ):
        temporal_embeddings, static_context = inputs

        flattened_embed = keras.ops.reshape(
            temporal_embeddings,
            [-1, self.time_steps, self.num_input_vars * self.d_model]
        )
        parallel_variables = keras.ops.reshape(
            temporal_embeddings, 
            [-1, self.time_steps, self.d_model]
        ) # tensor is shaped this way so that a GRN can be applied to each variable of each batch
        c_s = keras.ops.expand_dims(static_context, axis=1)

        # variable weights
        grn_outputs, _ = self.var_sel_weights(flattened_embed, c_s, training=training)
        variable_weights = self.softmax(grn_outputs)
        variable_weights = keras.ops.expand_dims(variable_weights, axis=2)

        # variable combination
        # transformed_embeddings = [
        #     grn_layer(temporal_embeddings[:, :, i, :], training=training)[0]
        #     for i, grn_layer in enumerate(self.grns_transformed_embeddings)
        # ]
        transformed_embeddings, _ = self.grns_transformed_embeddings(parallel_variables, training=training)
        transformed_embeddings = keras.ops.reshape(
            transformed_embeddings,
            [-1, self.time_steps, self.num_input_vars, self.d_model]
        )
        #transformed_embeddings = keras.ops.stack(transformed_embeddings, axis=2)
        temporal_vec = keras.ops.einsum('btij,btjk->btk', variable_weights, transformed_embeddings)
        return temporal_vec, keras.ops.squeeze(variable_weights, axis=2)

class TemporalFeatures(keras.Layer):
    def __init__(
        self, 
        d_model:int=16, # Embedding size, $d_\text{model}$
        dropout_rate:float=0., # Dropout rate
        **kwargs
    ):
        super(TemporalFeatures, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(TemporalFeatures, self).build(input_shape)
        self.hist_encoder = keras.layers.LSTM(
            units=self.d_model,
            return_sequences=True,
            return_state=True,
            stateful=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=self.dropout_rate,
            unroll=False,
            use_bias=True
        )
        self.fut_decoder = keras.layers.LSTM(
            units=self.d_model,
            return_sequences=True,
            return_state=False,
            stateful=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=self.dropout_rate,
            unroll=False,
            use_bias=True
        )
        self.lstm_glu = GatedLinearUnit(
            d_model=self.d_model, # Dimension of the GLU
            dropout_rate=self.dropout_rate, # Dropout rate
            use_time_distributed=True, # Apply the GLU across all time steps?
            activation=None # Activation function
        )
        self.add = keras.layers.Add()
        self.l_norm = keras.layers.LayerNormalization()

    def call(self, inputs, training=None):
        historical_features, future_features, c_h, c_c = inputs
        input_embeddings = keras.ops.concatenate(
            [historical_features, future_features],
            axis=1
        )

        history_lstm_encoder, state_h, state_c = self.hist_encoder(
            historical_features,
            initial_state=[
                c_h, # short-term state
                c_c  # long-term state
            ],
            training=training
        )

        future_lstm_decoder = self.fut_decoder(
            future_features,
            initial_state=[
                state_h, # short-term state
                state_c  # long-term state
            ],
            training=training
        )
        
        # this step concatenates at the time dimension, ie
        # the time series of history internal states are now
        # concated in sequence with the series of future internal states
        # $\phi(t,n) \in \{\phi(t,-k), \dots, \phi(t, \tau_{\text{max}})\}$
        lstm_layer = keras.ops.concatenate([history_lstm_encoder, future_lstm_decoder], axis=1)
        
        # Apply gated skip connection
        lstm_layer, _ = self.lstm_glu(
            lstm_layer,
            training=training
        )
        outputs = self.add([lstm_layer, input_embeddings])
        outputs = self.l_norm(outputs)
        # it's the temporal feature layer that is fed into the Temporal Fusion Decoder
        # its dimensions are (batch size / num time steps historical + future / hidden size)

        return outputs

class ScaledDotProductAttention(keras.Layer):
    def __init__(
        self,
        dropout_rate:float=0.0, # Will be ignored if `training=False`
        **kwargs
    ):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(ScaledDotProductAttention, self).build(input_shape)
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)
        self.activation = keras.layers.Activation('softmax')
        self.dot_22 = keras.layers.Dot(axes=(2, 2))
        self.dot_21 = keras.layers.Dot(axes=(2, 1))
        self.lambda_layer = keras.layers.Lambda(lambda x: (-1e9) * (1. - keras.ops.cast(x, 'float32')))
        self.add = keras.layers.Add()

    def call(
        self,
        q, # Queries, tensor of shape (?, time, D_model)
        k, # Keys, tensor of shape (?, time, D_model)
        v, # Values, tensor of shape (?, time, D_model)
        mask, # Masking if required (sets Softmax to very large value), tensor of shape (?, time, time)
        training=None, # Whether the layer is being trained or used in inference
    ):
        # returns Tuple (layer outputs, attention weights)
        scale = keras.ops.sqrt(keras.ops.cast(keras.ops.shape(k)[-1], dtype='float32'))
        attention = self.dot_22([q, k]) / scale
        #attention = keras.ops.einsum("bij,bjk->bik", q, keras.ops.transpose(k, axes=(0, 2, 1))) / scale
        if mask is not None:
            mmask = self.lambda_layer(mask)
            attention = self.add([attention, mmask])
        attention = self.activation(attention)
        if training:
            attention = self.dropout(attention)
        output = self.dot_21([attention, v])
        #output = keras.ops.einsum("btt,btd->bt", attention, v)
        return output, attention

class InterpretableMultiHeadAttention(keras.Layer):
    def __init__(
        self,
        n_head:int,
        d_model:int, # Embedding size, $d_\text{model}$
        dropout_rate:float, # Will be ignored if `training=False`
        **kwargs
    ):
        super(InterpretableMultiHeadAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_qk = self.d_v = d_model // n_head # the original model divides by number of heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(InterpretableMultiHeadAttention, self).build(input_shape)
        
        # using the same value layer facilitates interpretability
        vs_layer = keras.layers.Dense(self.d_v, use_bias=False, name="shared_value")

        # creates list of queries, keys and values across heads
        self.qs_layers = [keras.layers.Dense(self.d_qk) for _ in range(self.n_head)]
        self.ks_layers = [keras.layers.Dense(self.d_qk) for _ in range(self.n_head)]
        self.vs_layers = [vs_layer for _ in range(self.n_head)]

        self.attention = ScaledDotProductAttention(dropout_rate=self.dropout_rate)
        self.w_o = keras.layers.Dense(self.d_v, use_bias=False, name="W_v") # W_v in Eqs. (14)-(16), output weight matrix to project internal state to the original TFT
        self.dropout = keras.layers.Dropout(self.dropout_rate)

    def call(
        self,
        q, # Queries, tensor of shape (?, time, d_model)
        k, # Keys, tensor of shape (?, time, d_model)
        v, # Values, tensor of shape (?, time, d_model)
        mask=None, # Masking if required (sets Softmax to very large value), tensor of shape (?, time, time)
        training=None
    ):
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](q)
            vs = self.vs_layers[i](v)
           
            head, attn = self.attention(qs, ks, vs, mask, training=training)
            if training:
                head = self.dropout(head)
            heads.append(head)
            attns.append(attn)
        head = keras.ops.stack(heads) if self.n_head > 1 else heads[0]
        outputs = keras.ops.mean(head, axis=0) if self.n_head > 1 else head # H_tilde
        outputs = self.w_o(outputs)
        if training:
            outputs = self.dropout(outputs)
        return outputs, attn

@keras.saving.register_keras_serializable() # Make sure custom class can be saved with model.save()
class TFT(keras.Model):
    def __init__(
        self,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        d_model:int=16, # Embedding size, $d_\text{model}$
        output_size:int=1, # How many periods to nowcast/forecast?
        n_head:int=4,
        dropout_rate:float=0.1,
        skip_attention:bool=False, # Build a partial TFT without attention
        **kwargs
    ):
        super(TFT, self).__init__(**kwargs)
        self.quantiles = quantiles
        self.d_model = d_model
        self.output_size = output_size
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.skip_attention = skip_attention

    def build(self, input_shape):
        super(TFT, self).build(input_shape)
        
        self.input_layer = InputTFT(
            d_model=self.d_model,
            name="input"
        )
        self.svars = StaticVariableSelection(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name="static_variable_selection"
        )
        self.tvars_hist = TemporalVariableSelection(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name="historical_variable_selection"
        )
        self.tvars_fut = TemporalVariableSelection(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name="future_variable_selection"
        )
        self.static_context_s_grn = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="static_context_for_variable_selection"
        )
        self.static_context_h_grn = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="static_context_for_LSTM_state_h"
        )
        self.static_context_c_grn = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="static_context_for_LSTM_state_c"
        )
        self.static_context_e_grn = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="static_context_for_enrichment_of"
        )
        self.temporal_features = TemporalFeatures(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name="LSTM_encoder"
        )
        self.static_context_enrichment = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            name="static_context_enrichment"
        )
        if not self.skip_attention:
            self.attention = InterpretableMultiHeadAttention(
                n_head=self.n_head,
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                name="attention_heads"
            )
            self.attention_gating = GatedLinearUnit(
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                activation=None,
                name="attention_gating"
            )
            self.attn_grn = GatedResidualNetwork(
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                name="output_nonlinear_processing"
            )
            self.final_skip = GatedLinearUnit(
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                activation=None,
                name="final_skip_connection"
            )
            self.add = keras.layers.Add()
        self.l_norm = keras.layers.LayerNormalization()
        
        self.flat = keras.layers.Flatten(name="flatten")

        # Output layers:
        # In order to enforce monotoncity of the quantiles forecast only the lowest quantile
        # from a base forecast layer, and use output_len - 1 additional layers with ReLU activation
        # to produce the difference between the current quantile and the previous one
        output_len = len(self.quantiles)

        self.base_output_layer = keras.layers.TimeDistributed(
            keras.layers.Dense(1),
            name="output"
        )
        def elu_plus(x):
            return keras.activations.elu(x) + 1

        self.quantile_diff_layers = [
            keras.layers.TimeDistributed(
                keras.layers.Dense(1, activation=elu_plus),
                name=f"quantile_diff_{i}"
            ) 
            for i in range(output_len - 1)
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "quantiles": self.quantiles,
                "d_model": self.d_model,
                "output_size": self.output_size,
                "n_head": self.n_head,
                "dropout_rate" :self.dropout_rate,
                "skip_attention":  self.skip_attention,
            }
        )
        return config
    
    def _get_decoder_mask(
        self_attention_inputs # Inputs to the self-attention layer
    ):
        "Determines shape of decoder mask"
        len_s = keras.ops.shape(self_attention_inputs)[1] # length of inputs
        bs = keras.ops.shape(self_attention_inputs)[0] # batch shape
        mask = keras.ops.cumsum(keras.ops.eye(len_s), axis=0)

        ### warning: I had to manually implement some batch-wise shape here 
        ### because the new keras `eye` function does not have a batch_size arg.
        ### inspired by: https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/python/ops/linalg_ops_impl.py#L30
        ### <hack>
        mask = keras.ops.expand_dims(mask, axis=0)    
        mask = keras.ops.tile(mask, (bs, 1, 1))
        ### </hack>

        return mask

    def call(
        self,
        inputs,
        training=None
    ):
        "Creates the model architecture"
        
        # embedding the inputs
        cont_hist, cat_hist, cat_fut, cat_stat = inputs
        if len(cat_stat.shape) == 2:
            cat_stat = keras.ops.expand_dims(cat_stat, axis=-1)
            
        xi_hist, xi_fut, xi_stat = self.input_layer([cont_hist, cat_hist, cat_fut, cat_stat])

        # selecing the static covariates
        static_selected_vars, static_selection_weights = self.svars(xi_stat, training=training)

        # create context vectors from static data
        c_s, _ = self.static_context_s_grn(static_selected_vars, training=training) # for variable selection
        c_h, _ = self.static_context_h_grn(static_selected_vars, training=training) # for LSTM state h
        c_c, _ = self.static_context_c_grn(static_selected_vars, training=training) # for LSTM state c
        c_e, _ = self.static_context_e_grn(static_selected_vars, training=training) # for context enrichment of post-LSTM features

        # temporal variable selection
        hist_selected_vars, hist_selection_weights = self.tvars_hist(
            [xi_hist, c_s],
            training=training
        )
        fut_selected_vars, fut_selection_weights = self.tvars_fut(
            [xi_fut, c_s],
            training=training
        )
        input_embeddings = keras.ops.concatenate(
            [hist_selected_vars, fut_selected_vars],
            axis=1
        )

        features = self.temporal_features(
            [hist_selected_vars, fut_selected_vars, c_h, c_c],
            training=training
        )
        
        # static context enrichment
        enriched, _ = self.static_context_enrichment(
            features, 
            additional_context=keras.ops.expand_dims(c_e, axis=1),
            training=training
        )
        if not self.skip_attention:
            mask = self._get_decoder_mask(enriched)
            attn_output, self_attn = self.attention(
                q=enriched,
                k=enriched,
                v=enriched,
                mask=mask,
                training=training
            )
            attn_output, _ = self.attention_gating(attn_output)
            output = self.add([enriched, attn_output])
            output = self.l_norm(output)
            output, _ = self.attn_grn(output)
            output, _ = self.final_skip(output)
            output = self.add([features, output])
        else:
            output = enriched
        output = self.l_norm(output)
        
        # Base quantile output
        base_output = output[Ellipsis,hist_selected_vars.shape[1]:,:]
        base_quantile = self.base_output_layer(base_output)
                
        # Additional layers for remaining quantiles
        quantile_outputs = [base_quantile]
        for i in range(len(self.quantiles) - 1):
            quantile_diff = self.quantile_diff_layers[i](base_output)
            quantile_output = quantile_outputs[-1] + quantile_diff
            quantile_outputs.append(quantile_output)
        final_output = keras.ops.concatenate(quantile_outputs, axis=-1)
        return final_output