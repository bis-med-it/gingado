# Contributing to `gingado`

Welcome, and thank you for your interest in contributing to `gingado`! Whether it's reporting issues, suggesting new features, or contributing to the code, documentation, or tests, your involvement is highly appreciated.

## Getting Started

### Setting Up for Development

To get started with contributing to `gingado`, you need to set up your development environment. This includes installing necessary tools and configuring your system to work efficiently with our codebase.

To install the dependencies to work with gingado, you need to install the execution dependencies and the development dependencies of gingado:

```
pip install -r requirements.txt
pip install -r dev_requirements.txt
```

To work with the documentation an tests, we use `quarto`. You can either use RStudio or Visual Studio Code with the Quarto extension installed for editing `.qmd` files which are used for generating documentation. Here's how you can set up your environment:

1. **Install Quarto**: If you haven't already, install Quarto from [Quarto's official website](https://quarto.org/docs/get-started/). Follow the instructions for your operating system.
2. **Configure Your Editor**:
   - **For RStudio**: Quarto is integrated with RStudio. Ensure you have the latest version of RStudio to work with Quarto seamlessly.
   - **For Visual Studio Code**: Install the Quarto extension from the Visual Studio Code marketplace. This extension provides support for `.qmd` files, including syntax highlighting and preview capabilities.

### Reporting Issues and Suggestions

If you encounter a bug, have suggestions, or want to propose new functionalities:

- **Check Existing Issues**: Ensure the bug or suggestion hasn't been reported/mentioned before by searching under [Issues](https://github.com/bis-med-it/gingado/issues) on GitHub.
- **Create a New Issue**: If no existing issue addresses the problem or suggestion, please [create a new issue](https://github.com/bis-med-it/gingado/issues), providing a descriptive title, a clear description, and as much relevant information as possible. For bugs, include a code sample or an executable test case demonstrating the expected behavior that is not occurring, along with complete error messages.

### Contributing Code

#### Changes to Codebase

To contribute changes to the codebase, including documentation and tests, follow these guidelines:

- **Document New Features**: Clearly document new functions or classes in the `.qmd` files. Write clear descriptions, specify expected inputs and outputs, and include any relevant information to understand the functionality.
- **Include Tests**: Implement tests for new functionalities as part of the `.qmd` files to ensure the integrity and reliability of the code. Make sure your tests cover the expected behavior and edge cases.

#### Pull Request (PR) Guidelines

- **Focused PRs**: Each PR should be focused on a single topic. Avoid combining unrelated changes.
- **Separate Style and Functional Changes**: Do not mix style changes with functional changes in the same PR.
- **Preserve File Style**: Avoid adding or removing vertical whitespace unnecessarily. Keep the original style of the files you edit.
- **Development Process**: Do not use a submitted PR as a development playground. If additional work is needed, consider closing the PR, completing the work, and then submitting a new PR.
- **Responding to Feedback**: If your PR requires changes based on feedback, continue committing to the same PR unless the changes are substantial. In that case, it might be better to start a new PR.

### Documentation Contributions

When contributing to the documentation, ensure your contributions are made within `.qmd` files. This is essential for the changes to be correctly reflected in the generated documentation through Quarto.

## Your Contributions Make a Difference

By contributing to `gingado`, you are part of a community that values collaboration, innovation, and learning. We look forward to your contributions and are excited to see what we can build together.
