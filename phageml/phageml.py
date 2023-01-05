#!/usr/bin/env python3
# Copyright 2023 James W Gillespie, Ph.D.

import argparse


def parse_args(args) -> argparse.Namespace:
    '''PhageML Command Line Parser'''
    parser = argparse.ArgumentParser(prog="phageml")

    return parser.parse_args(args)
