# Coding Standards 

To maintain readability, here are the coding standards for the scMKL source 
code.

## Contents

1) Module layout and formatting
2) Basic operations
3) Function parameters and arguments
4) Doc-strings
5) Module Structure


### Module Layout and Formatting

No lines of code should exceed 79 characters for readability.

Doc-strings should not exceed 72 characters.

Every module should start with external package imports followed by a line 
and local package imports, then two lines:

```python
import scipy
import numpy as np

from sckml.data_processing import process_data


def my_func(a, b):
    '''
    Sums `a` and `b`.
    '''
    return a + b
```

Every function in a module should be separated by two empty lines.

Functions should not exceed 50 lines of code. 30 lines or less is prefered to 
keep code modular and readable.


### Basic Operations

Addition and subtraction operators should be surrounded by spaces:
```python
# Wrong
result = a+b

# Right
result = a + b
```

Variable assignment (`=`) should also be surrounded by spaces:
```python
# Wrong
result=a + b
```
```python
# Right
result = a + b
```

Multiplication, exponent, and division operators should not be surrounded by 
spaces:
```python
# Wrong
result = a / b
```
```python
# Right
result = a/b
```


### Function parameters and arguments

When parameterizing a function there are two things that should be done:

1) Variable typing
    
    When defining a function and object type is important, use type 
    annotations. In the example below, if the input array is a python list,
    the function will fail.
    ```python
    # Wrong
    def my_func(array):
        '''
        Returns the indices of an array.
        '''
        n_samples = array.shape[0]
        indices = np.arange(n_samples)

        return indices
    ```
    To prevent this and include this important information in the 
    documentation, the function should be written as:
    ```python
    # Right
    def my_func(array: np.ndarray):
        '''
        Returns the indices of an array.

        Parameters
        ----------
        array : np.ndarray
            > An array where dimension 0 corresponds to samples.
        '''
        n_samples = array.shape[0]
        indices = np.arange(n_samples)

        return indices
    ```

2) Default assignment spacing

    When a function has a parameter with a default argument, the `=` should 
    not be surrounded by spaces:
    ```python
    def my_func(a, b=10)
        '''
        Sums `a` and `b`.
        '''
        result = a + b
        return result
    ```

When calling a function and assigning arguments, `=` should not be surrounded 
by spaces and each argument should be followed by a space:
```python
# Wrong
result = my_func(a = 10,b = 5)
```
```python
# Right
result = my_func(a=10, b=5)
```


### Doc-strings

Doc-strings should include the following separated by a space:

    - Brief description
    - Parameters with definitions and types
    - Return variables with definitions and types
    - Examples (if top-level function)

Doc-string lines should not exceed 72 characters. When using a variable's 
name, the name should be surrounded by backticks (i.e. `my_variable`). All 
parameters should be listed with types and definitions. Return values should 
be listed the same as parameters. If the function is designed to be called by 
external users, examples must be included.

Poorly written and unformatted doc-string:
```python
# Wrong
def my_func(a, b):
    """
    a plus b with a and b being integers or floats.
    """
    result = a + b
    return result
```
Well written and formatted doc-string:
```python
def my_func(a, b):
    """
    Returns the sum of a and b.

    Parameters
    ----------
    a : int | float
        > A number to sum with `b`.

    b : int | float
        > A number to sum with `a`.

    Returns
    -------
    result : int | float
        > The sum of `a` and `b`.

    Examples
    --------
    >>> a = 5
    >>> b = 7
    >>>
    >>> my_func(a, b)
    12
    """
```
