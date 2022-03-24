#!/usr/bin/env python3


class TreeConfig(object):

    max_level = 6

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance


TC = TreeConfig
