# Contributing to Fair Client Selection in Federated Learning

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the codebase.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Adding New Algorithms](#adding-new-algorithms)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites

- Python >= 3.8
- Git
- Basic understanding of Federated Learning and fairness concepts

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/your-username/CS6007-proj.git
cd CS6007-proj

# Add upstream remote
git remote add upstream https://github.com/abhaychandra123/CS6007-proj.git
```

## 💻 Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies (if exists)
```

### 3. Verify Setup

```bash
cd fair_client_selection
python -c "from algorithms import *; from utils import *; print('✓ Setup successful')"
python fast_selection_benchmark.py  # Run quick test
```

## 📝 Code Style Guidelines

### Python Style

We follow **PEP 8** with some specific conventions:

1. **Formatting**: Use `black` with 100-character line length
2. **Import Order**: stdlib → third-party → local (use `isort`)
3. **Type Hints**: Required for all function signatures
4. **Docstrings**: Google style for all public functions/classes

### Example

```python
"""Module for implementing client selection algorithms."""
import logging
from typing import List, Dict, Optional

import numpy as np
import torch

from .base import FairClientSelector

logger = logging.getLogger(__name__)


class MyAlgorithm(FairClientSelector):
    """Short one-line description.
    
    Longer description explaining the algorithm, its purpose,
    and key characteristics.
    
    Args:
        num_clients: Total number of clients in the system
        clients_per_round: Number of clients to select each round
        param1: Description of parameter 1
        seed: Random seed for reproducibility
    
    Raises:
        ValueError: If num_clients <= 0 or clients_per_round > num_clients
    
    Example:
        >>> selector = MyAlgorithm(num_clients=100, clients_per_round=10)
        >>> selected = selector.select_clients(client_info, round_num=0)
    """
    
    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        param1: float = 0.5,
        seed: int = 42
    ) -> None:
        super().__init__(num_clients, clients_per_round, seed)
        
        # Input validation
        if param1 < 0 or param1 > 1:
            raise ValueError(f"param1 must be in [0, 1], got {param1}")
        
        self.param1 = param1
        logger.info(f"Initialized {self.__class__.__name__} with param1={param1}")
    
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """Select clients for the current round.
        
        Args:
            client_info: Dictionary mapping client_id to client metadata
            round_num: Current training round number
        
        Returns:
            List of selected client IDs
        
        Raises:
            ValueError: If client_info is empty
        """
        if not client_info:
            raise ValueError("client_info cannot be empty")
        
        # Your selection logic here
        selected = []
        
        # Always call this at the end
        self.update_selection_history(selected)
        return selected
```

### Error Handling

- **Input Validation**: Validate all inputs at function entry
- **Logging**: Use logging module, not print statements
- **Exceptions**: Raise appropriate exceptions with clear messages
- **Edge Cases**: Handle empty lists, zero values, None, etc.

```python
# Good
if not client_info:
    raise ValueError("client_info cannot be empty")

# Bad
assert client_info, "client_info cannot be empty"  # Don't use assert for validation
```

### Logging Levels

```python
logger.debug("Detailed information for debugging")  # Development only
logger.info("General information about execution")   # Normal operation
logger.warning("Something unexpected but recoverable") # Potential issues
logger.error("Error occurred but program continues")   # Errors
logger.critical("Critical error, program may crash")   # Serious problems
```

## 🧮 Adding New Algorithms

### Step 1: Create Algorithm File

```bash
cd fair_client_selection/algorithms
touch my_algorithm.py
```

### Step 2: Implement Base Class

```python
from .base import FairClientSelector
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MyAlgorithm(FairClientSelector):
    """Your algorithm implementation."""
    
    def __init__(self, num_clients: int, clients_per_round: int, **kwargs):
        super().__init__(num_clients, clients_per_round, kwargs.get('seed', 42))
        # Initialize your parameters
    
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """Implement selection logic."""
        # Validate inputs
        if not client_info:
            raise ValueError("client_info cannot be empty")
        
        # Your selection algorithm
        selected = []  # Your logic here
        
        # Update history (required)
        self.update_selection_history(selected)
        return selected
```

### Step 3: Add to __init__.py

```python
# In fair_client_selection/algorithms/__init__.py
from .my_algorithm import MyAlgorithm

__all__ = [
    'FairClientSelector',
    'RandomSelector',
    # ... existing algorithms
    'MyAlgorithm',  # Add here
]
```

### Step 4: Add to Benchmark

```python
# In fast_selection_benchmark.py
from algorithms.my_algorithm import MyAlgorithm

algorithms = {
    # ... existing algorithms
    'MyAlgo': MyAlgorithm(NUM_CLIENTS, CLIENTS_PER_ROUND, **params),
}
```

## 🧪 Testing Requirements

### Unit Tests

Create test file `test_my_algorithm.py`:

```python
import unittest
import numpy as np
from algorithms.my_algorithm import MyAlgorithm


class TestMyAlgorithm(unittest.TestCase):
    def setUp(self):
        self.num_clients = 20
        self.clients_per_round = 5
        self.selector = MyAlgorithm(self.num_clients, self.clients_per_round)
    
    def test_initialization(self):
        """Test algorithm initializes correctly."""
        self.assertEqual(self.selector.num_clients, 20)
        self.assertEqual(self.selector.clients_per_round, 5)
    
    def test_select_clients(self):
        """Test client selection."""
        client_info = {i: {'data': np.random.randn(10)} for i in range(20)}
        selected = self.selector.select_clients(client_info, round_num=0)
        
        # Verify selection size
        self.assertEqual(len(selected), 5)
        
        # Verify all selected clients are valid
        for client_id in selected:
            self.assertIn(client_id, client_info)
            self.assertGreaterEqual(client_id, 0)
            self.assertLess(client_id, self.num_clients)
    
    def test_empty_client_info(self):
        """Test handling of empty client_info."""
        with self.assertRaises(ValueError):
            self.selector.select_clients({}, round_num=0)
    
    def test_fairness_over_rounds(self):
        """Test fairness improves over multiple rounds."""
        client_info = {i: {'data': np.random.randn(10)} for i in range(20)}
        
        for round_num in range(30):
            self.selector.select_clients(client_info, round_num)
        
        stats = self.selector.get_selection_statistics()
        
        # Check basic fairness properties
        self.assertGreater(stats['mean_selections'], 0)
        self.assertLess(stats['cv_selections'], 1.0)  # Adjust threshold as needed


if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```bash
# Run fast benchmark to ensure integration
cd fair_client_selection
python fast_selection_benchmark.py
```

### Required Tests

Before submitting a PR, ensure:

- [ ] Unit tests for all new functions
- [ ] Edge case handling (empty inputs, boundary values)
- [ ] Integration test passes (fast_selection_benchmark.py)
- [ ] No decrease in existing algorithm performance
- [ ] Documentation is complete

## 📤 Submitting Changes

### 1. Create Feature Branch

```bash
git checkout -b feature/my-new-algorithm
```

### 2. Make Changes

- Write code following style guidelines
- Add tests
- Update documentation

### 3. Commit Changes

```bash
git add .
git commit -m "feat: Add MyAlgorithm for fairness-aware selection

- Implements novel selection strategy based on X
- Achieves Y% improvement in fairness metric Z
- Adds comprehensive tests and documentation
- Closes #123"
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/my-new-algorithm
```

Then create Pull Request on GitHub with:

- **Title**: Clear, descriptive title
- **Description**: 
  - What changes were made
  - Why they were made
  - How to test
  - Related issues
- **Checklist**: Mark completed items

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass
```

## 📚 Documentation

### Code Documentation

- **Modules**: Docstring at top of file
- **Classes**: Docstring with Args, Raises, Example
- **Functions**: Docstring with Args, Returns, Raises
- **Complex Logic**: Inline comments explaining why, not what

### Algorithm Documentation

When adding a new algorithm, update:

1. **Algorithm docstring** with paper reference
2. **README.md** with algorithm in table
3. **report.tex** if adding to research paper
4. **Example usage** in docstring or separate example file

### Example Documentation

```python
class MyAlgorithm(FairClientSelector):
    """Client selection using novel fairness approach.
    
    Based on: Author et al. (2024) - "Paper Title"
    URL: https://arxiv.org/abs/XXXX.XXXXX
    
    This algorithm improves upon existing approaches by...
    Key innovation: ...
    
    Time Complexity: O(N log N)
    Space Complexity: O(N)
    
    Args:
        num_clients: Total number of clients
        clients_per_round: Clients selected per round
        lambda_param: Trade-off parameter (default: 0.5)
        seed: Random seed for reproducibility
    
    Raises:
        ValueError: If num_clients <= 0
        ValueError: If lambda_param not in [0, 1]
    
    Example:
        >>> selector = MyAlgorithm(num_clients=100, clients_per_round=10)
        >>> client_info = {i: {'embedding': np.random.randn(10)} for i in range(100)}
        >>> selected = selector.select_clients(client_info, round_num=0)
        >>> print(f"Selected: {selected}")
        Selected: [3, 15, 42, 67, 89, 91, 23, 56, 78, 34]
    
    References:
        [1] Author, A. et al. (2024). Paper Title. Conference/Journal.
    """
```

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. With parameters '...'
3. See error

**Expected behavior**
What you expected to happen

**Actual behavior**
What actually happened

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- PyTorch version: [e.g., 1.12.0]

**Additional context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution you'd like**
Clear description of desired solution

**Describe alternatives you've considered**
Other solutions considered

**Additional context**
Any other relevant information
```

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Email**: For private concerns

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in research paper (if applicable)
- Credited in release notes

Thank you for contributing to Fair Client Selection in Federated Learning!
