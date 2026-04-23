from numpy import array, sqrt, matmul, allclose


state : array
"""Contains the current state of the system."""


def init() -> None:
    """Initiales the state with classical possibility 0.
    (I.e., vector (1,0).) """
    ...


def x() -> None:
    """Applies the X-gate to the current state."""
    ...


def y() -> None:
    """Applies the Y-gate to the current state."""
    ...


def z() -> None:
    """Applies the Z-gate to the current state."""
    ...


def h() -> None:
    """Applies the Hadamard-gate to the current state."""
    ...

def s() -> None:
    """Applies the S-gate to the current state."""
    ...


def current_state() -> array:
    ...


def main() -> None:
    init()
    h()
    z()
    h()
    result = current_state()
    print(result)
    assert allclose(result, array([0, 1])), result


if __name__ == '__main__':
    main()
