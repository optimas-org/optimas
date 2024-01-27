"""Contains the definition of the Task class for multitask optimization."""

from pydantic import BaseModel


class Task(BaseModel):
    """Defines a task to be used in multitask optimization.

    Parameters
    ----------
    name : str
        Name of the task.
    n_init : int
        Number of task evaluations to perform in the initialization batch.
    n_opt : int
        Number of task evaluations to perform per optimization batch.

    """

    name: str
    n_init: int
    n_opt: int

    def __init__(self, name: str, n_init: int, n_opt: int) -> None:
        super().__init__(name=name, n_init=n_init, n_opt=n_opt)
