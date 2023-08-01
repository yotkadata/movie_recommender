"""
Utility functions for the project.
"""

import cProfile
import io
import pstats


def profile(fnc):
    """
    A decorator that uses cProfile to profile a function.
    Starts the profile before executing a function,
    then executes the function, then stops the profile,
    and finally prints out a diagnostics report.
    """

    def inner(*args, **kwargs):
        # Initialize the profiler
        prof = cProfile.Profile()

        # Start the profiler
        prof.enable()

        # Execute the function
        retval = fnc(*args, **kwargs)

        # Stop the profiler
        prof.disable()

        # Print the results to the standard output
        string_io = io.StringIO()
        p_stats = pstats.Stats(prof, stream=string_io).sort_stats("cumulative")
        p_stats.print_stats()
        print(string_io.getvalue())

        # Return the actual return value of the inner function we executed
        return retval

    # Execute the inner function
    return inner
