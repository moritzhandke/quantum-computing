from typing import Iterable

import numpy
from numpy import array, matmul


size = 5
"""Number of deterministic possibilities.
That is, all distributions are over {0,...,size-1}.
Your code should work for other values of `size`, too."""


def example_process_1(input: int) -> array:
    """On input `input`, returns uniform distribution over
    {0,...,input-1}."""
    return array([1/(input+1) if i <= input else 0
                  for i in range(0,size)])


def example_process_2(input: int) -> array:
    """On input `input`, returns a distribution that assigned probability 1
    to `(input + 1) % size`"""
    output = (input + 1) % size
    return array([0]*output + [1] + [0]*(size-output-1))


def join_rows(rows: Iterable[array]) -> array:
    """Given a list of rows, makes a matrix with those rows"""
    return array(list(rows))


def join_columns(columns: Iterable[array]) -> array:
    """Given a list of columns, makes a matrix with those columns"""
    return join_rows(columns).transpose()


def matrix_from_probabilistic_process(process) -> array:
    """Takes a probabilistic process `process` and returns a matrix describing
    it.

    Here `process` is a function that takes an int (from 0,...,size-1)
    and returns a probability distribution (as a vector) for the output.
    """
    return join_columns(process(i) for i in range(size))


def apply(process, distrib: array) -> array:
    """Return the new distribution after applying the probabilistic process `process`
    to the distribution `distrib`."""
    matrix = matrix_from_probabilistic_process(process)
    return matmul(matrix, distrib)


def example_distribution() -> array:
    """Returns a probability distribution that assigns
    probability 1 to `size-1`."""
    return array([0]*(size-1) + [1])


def main():
    """Main program that tests your functions.
    Note: this is not the actual test case that will be used later.
    """
    assert size == 5  # The hardcoded expected values below only work for size=5
    state1 = example_distribution()
    result1 = apply(example_process_1, state1)
    assert numpy.array_equal(result1, [1/5, 1/5, 1/5, 1/5, 1/5]), result1
    state2 = example_distribution()
    result2 = apply(example_process_2, state2)
    assert numpy.array_equal(result2, [1/3*3, 0, 0, 0, 0]), result2
    print("All tests passed!")

if __name__ == '__main__':
    main()
