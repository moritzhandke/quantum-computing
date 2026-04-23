from numpy import array, sqrt, matmul, allclose


state : array
"""Contains the current state of the system."""


def init() -> None:
    """Initiales the state with classical possibility 0.
    (I.e., vector (1,0).) """
    global state
    state = array([1, 0])


def x() -> None:
    """Applies the X-gate to the current state.
      | 0 1 |
      | 1 0 |
    """
    global state
    x_gate = array([[0, 1], [1, 0]]) 
    state = matmul(x_gate, state)


def y() -> None:
    """Applies the Y-gate to the current state.
      | 0 -i |
      | i  0 |
    """
    global state
    y_gate = array([[0, -1j], [1j, 0]])
    state = matmul(y_gate, state)


def z() -> None:
    """Applies the Z-gate to the current state.
      | 1  0 |
      | 0 -1 |
    """
    global state
    z_gate = array([[1, 0], [0, -1]])
    state = matmul(z_gate, state)


def h() -> None:
    """Applies the Hadamard-gate to the current state.
      | 1  1 |
      | 1 -1 |
    """
    global state
    h_gate = array([[1, 1], [1, -1]]) / sqrt(2)
    state = matmul(h_gate, state)


def s() -> None:
    """Applies the S-gate to the current state.
        | 1 0 |
        | 0 i |
    """
    global state
    s_gate = array([[1, 0], [0, 1j]])
    state = matmul(s_gate, state)
    return state


def current_state() -> array:
    return state


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
