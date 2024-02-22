from __future__ import annotations # allows multiple typing of arguments in Python versions prior to 3.10

import copy
import json
from .utils import get_datetime

__all__ = ['ggdModelDocumentation', 'ModelCard', 'ForecastCard']

class ggdModelDocumentation:
    "Base class for gingado Documenters"

    def setup_template(self):
        "Set up the template from the JSON documentation"
        self.json_doc = copy.deepcopy(self.__class__.template)
        for k in self.json_doc.keys():
            self.json_doc[k].pop('field_description', "")

    def show_template(
        self, 
        indent:bool=True
        ):
        """
        Show documentation template in JSON format.

        Args:
            indent (bool): Whether to print JSON documentation template with indentation for easier human reading.
        """
        if indent:
            print(json.dumps(self.__class__.template, indent=self.indent_level))
        else:
            return self.__class__.template
        
    def documentation_path(self):
        "Show path to documentation"
        print(self.file_path)

    def show_json(self):
        "Show documentation in JSON format"
        return self.json_doc
        #print(json.dumps(self.json_doc, indent=self.indent_level))

    def save_json(
        self, 
        file_path:str
        ):
        """
        Save the documentation in JSON format in the specified file.

        Args:
            file_path (str): Path to save JSON file.
        """
        with open(file_path, 'w') as f:
            json.dump(self.json_doc, f)

    def read_json(
        self, 
        file_path:str
        ):
        """
        Load documentation JSON from path.

        Args:
            file_path (str): Path to JSON file or path defined in `file_path` if None.
        """
        if file_path is None:
            file_path = self.file_path
        f = open(file_path)
        self.json_doc = json.load(f)

    def open_questions(self):
        "List open fields in the documentation"
        return [
                    k + "__" + v 
                    for k, v in self.json_doc.items()
                    if isinstance(v, dict)
                    for v, i in v.items()
                    if i == self.__class__.template[k][v]
        ]

    def fill_info(
        self, 
        new_info:dict
        ):
        """
        Include information in model documentation.

        Args:
            new_info (dict): Dictionary with information to be added to the model documentation.
        """
        for k, v in new_info.items():
            if k not in self.__class__.template.keys():
                raise KeyError(f"key '{k}' is not in the documentation template. The template's keys are: {self.__class__.template.keys()}")
            if isinstance(v, dict) and isinstance(self.json_doc[k], dict):
                for v_k, v_v in v.items():
                    if v_k == 'field_description':
                        raise KeyError("The key 'field_description' is not supposed to be changed from the template definition.")
                    if v_k not in self.json_doc[k].keys():
                        field_keys = [k for k in self.__class__.template[k].keys() if k != 'field_description']
                        raise KeyError(f"key '{v_k}' is not in the documentation template's item {k}. These template item's keys are: {field_keys}")
                    #self.json_doc[k][v_k] = v_v
                    for kk in self.json_doc[k].keys():
                        if kk == v_k:
                            self.json_doc[k][kk] = v_v
            else:
                self.json_doc.update({k: v})

    def _read_attr(
        self, 
        model
        ):
        "For use of method `read_model`"
        for a in dir(model):
            if a == '_estimator_type' or a.endswith("_") and not a.startswith("_") and not a.endswith("__"):
                try:
                    model_attr = model.__getattribute__(a)
                    yield {a: model_attr}
                except:
                    pass

    def read_model(
        self, 
        model
        ):
        """
        Read automatically information from the model and add to documentation.

        Args:
            model: The model to be documented.
        """
        if "keras" in str(type(model)):
            model_info = model.to_json()
        else:
            model_info = list(self._read_attr(model))
            model_info = {k:v for i in model_info for k, v in i.items()}
        self.fill_model_info(model_info)

    def fill_model_info(
        self, 
        model_info:str|dict,
        model_info_keyname:str='model_details'
        ):
        """
        Called automatically, or by the user, to add model information to the documentation according to its template.

        Args:
            model_info (str | dict): Information about the model to be added in the documentation.
            model_info_keyname (str): Dictionary key in the Documenter template to which this information should be linked.
        """
        model_info_template = {model_info_keyname: {'info': model_info}}
        self.fill_info(model_info_template)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return json.dumps(self.json_doc, indent=4)

    def __repr__(self):
        return f"{self.__class__}()"


class ModelCard(ggdModelDocumentation):
    "A gingado Documenter based on @ModelCards"
    template = {
        "model_details": {
            "field_description": "Basic information about the model",
            "developer": "Person or organisation developing the model",
            "datetime": "Model date",
            "version": "Model version",
            "type": "Model type",
            "info": "Information about training algorithms, parameters, fairness constraints or other applied approaches, and features",
            "paper": "Paper or other resource for more information",
            "citation": "Citation details",
            "license": "License",
            "contact": "Where to send questions or comments about the model"
        },
        "intended_use": {
            "field_description": "Use cases that were envisioned during development",
            "primary_uses": "Primary intended uses",
            "primary_users": "Primary intended users",
            "out_of_scope": "Out-of-scope use cases"
        },
        "factors": {
            "field_description": "Factors could include demographic or phenotypic groups, environmental conditions, technical attributes, or others",
            "relevant": "Relevant factors",
            "evaluation": "Evaluation factors" 
        },
        "metrics": {
            "field_description": "Metrics should be chosen to reflect potential real world impacts of the model",
            "performance_measures": "Model performance measures",
            "thresholds": "Decision thresholds",
            "variation_approaches": "Variation approaches"
        },
        "evaluation_data": {
            "field_description": "Details on the dataset(s) used for the quantitative analyses in the documentation",
            "datasets": "Datasets",
            "motivation": "Motivation",
            "preprocessing": "Preprocessing"
        },
        "training_data": {
            "field_description": "May not be possible to provide in practice. When possible, this section should mirror 'Evaluation Data'. If such detail is not possible, minimal allowable information should be provided here, such as details of the distribution over various factors in the training datasets.",
            "training_data": "Information on training data"
        },
        "quant_analyses": {
            "field_description": "Quantitative Analyses",
            "unitary": "Unitary results",
            "intersectional": "Intersectional results"
        },
        "ethical_considerations": {
            "field_description": "Ethical considerations that went into model development, surfacing ethical challenges and solutions to stakeholders. Ethical analysis does not always lead to precise solutions, but the process of ethical contemplation is worthwhile to inform on responsible practices and next steps in future work.",
            "sensitive_data": "Does the model use any sensitive data (e.g., protected classes)?",
            "human_life": "Is the model intended to inform decisions about matters central to human life or flourishing - e.g., health or safety? Or could it be used in such a way?",
            "mitigations": "What risk mitigation strategies were used during model development?",
            "risks_and_harms": "What risks may be present in model usage? Try to identify the potential recipients,likelihood, and magnitude of harms. If these cannot be determined, note that they were considered but remain unknown",
            "use_cases": "Are there any known model use cases that are especially fraught?",
            "additional_information": "If possible, this section should also include any additional ethical considerations that went into model development, for example, review by an external board, or testing with a specific community."
        },
        "caveats_recommendations": {
            "field_description": "Additional concerns that were not covered in the previous sections",
            "caveats": "For example, did the results suggest any further testing? Were there any relevant groups that were not represented in the evaluation dataset?",
            "recommendations": "Are there additional recommendations for model use? What are the ideal characteristics of an evaluation dataset for this model?"
        }
    }

    def __init__(
        self,
        file_path:str="",
        autofill:bool=True,
        indent_level:int|None=2
        ):
        """Initializes the ModelCard with specified settings.

        Args:
            file_path: Path for the JSON file with the documentation.
            autofill: Whether the Documenter object should autofill when created.
            indent_level: Level of indentation during serialization to JSON.
        """
        self.file_path = file_path
        self.autofill = autofill
        self.indent_level = indent_level
        self.setup_template()
        if self.autofill:
            self.autofill_template()            

    def autofill_template(self):
        "Create an empty model card template, then fills it with information that is automatically obtained from the system"
        auto_info = {
            'model_details': {
                'datetime': get_datetime()
            }
        }
        self.fill_info(auto_info)

class ForecastCard(ggdModelDocumentation):
    "A gingado Documenter for forecasting or nowcasting use cases"
    template = {
        "model_details": {
            "field_description": "Basic information about the model",
            "variable": "Variable(s) being forecasted or nowcasted",
            "jurisdiction": "Jurisdiction(s) of the variable being forecasted or nowcasted",
            "developer": "Person or organisation developing the model",
            "datetime": "Model date",
            "version": "Model version",
            "type": "Model type",
            "pipeline": "Description of the pipeline steps being used",
            "info": "Information about training algorithms, parameters, fairness constraints or other applied approaches, and features",
            "econometric_model": "Information about the econometric model or technique",
            "paper": "Paper or other resource for more information",
            "citation": "Citation details",
            "license": "License",
            "contact": "Where to send questions or comments about the model"
        },
        "intended_use": {
            "field_description": "Use cases that were envisioned during development",
            "primary_uses": "Primary intended uses",
            "primary_users": "Primary intended users",
            "out_of_scope": "Out-of-scope use cases"
        },
        "factors": {
            "field_description": "Factors could include demographic or phenotypic groups, environmental conditions, technical attributes, or others",
            "relevant": "Relevant factors",
            "evaluation": "Evaluation factors" 
        },
        "metrics": {
            "field_description": "Metrics should be chosen to reflect potential real world impacts of the model",
            "performance_measures": "Model performance measures",
            "estimation_approaches": "How are the evaluation metrics calculated? Include information on the cross-validation approach, if used"
        },
        "data": {
            "field_description": "Details on the dataset(s) used for the training and evaluation of the model",
            "datasets": "Datasets",
            "preprocessing": "Preprocessing",
            "cutoff_date": "Cut-off date that separates training from evaluation data"
        },
        "ethical_considerations": {
            "field_description": "Ethical considerations that went into model development, surfacing ethical challenges and solutions to stakeholders. Ethical analysis does not always lead to precise solutions, but the process of ethical contemplation is worthwhile to inform on responsible practices and next steps in future work.",
            "sensitive_data": "Does the model use any sensitive data (e.g., protected classes)?",
            "risks_and_harms": "What risks may be present in model usage? Try to identify the potential recipients, likelihood, and magnitude of harms. If these cannot be determined, note that they were considered but remain unknown",
            "use_cases": "Are there any known model use cases that are especially fraught?",
            "additional_information": "If possible, this section should also include any additional ethical considerations that went into model development, for example, review by an external board, or testing with a specific community."
        },
        "caveats_recommendations": {
            "field_description": "Additional concerns that were not covered in the previous sections",
            "caveats": "For example, did the results suggest any further testing? Were there any relevant groups that were not represented in the evaluation dataset?",
            "recommendations": "Are there additional recommendations for model use? What are the ideal characteristics of an evaluation dataset for this model?"
        }
    }

    def __init__(self,
        file_path:str="",
        autofill:bool=True,
        indent_level:int|None=2
        ):
        """Initializes the ForecastCard object with options for file path, autofill, and JSON indentation.

        Args:
            file_path (str): Path for the JSON file with the documentation. Defaults to an empty string.
            autofill (bool): Whether the Documenter object should autofill template sections upon creation. Defaults to True.
            indent_level (int | None): Level of indentation during serialization to JSON. Defaults to 2. Use `None` for compact JSON.
        """
        self.file_path = file_path
        self.autofill = autofill
        self.indent_level = indent_level
        self.setup_template()
        if self.autofill:
            self.autofill_template()            

    def autofill_template(self):
        "Create an empty model card template, then fills it with information that is automatically obtained from the system"
        auto_info = {
            'model_details': {
                'datetime': get_datetime()
            }
        }
        self.fill_info(auto_info)
