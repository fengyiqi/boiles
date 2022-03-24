#!/usr/bin/env python3


class OptProblems:
    r"""
    A class that defines the optimization problems. This is a singleton class. If user needs to update these
    configurations, please import this class at the beginning of the script and assign corresponding configurations.

    test_cases: a list contains the information for calculating the objective function, see "test_cases" folder for
                more details;
    ref_point: reference point. It is only needed for multi-objective optimization;
    """

    test_cases = None
    ref_point = None

    # Singleton mode
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance


OP = OptProblems
