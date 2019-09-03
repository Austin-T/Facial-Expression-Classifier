"""Here is the main file for running the "Facial Expression Recognition" program.
This file must be run as the entry point to the program. It serves to create an
'ExpressionAnalyst' object, which will execute the program internally after a call
to the 'run()' method has been made on the object.

Version: 1
Date: 2019-09-02
Contributors: Austin Tralnberg
"""

from expression_analyst import ExpressionAnalyst


def main():

    # Create 'ExpressionAnalyst object
    analyst = ExpressionAnalyst()

    # Run the program
    analyst.run()


if __name__ == '__main__':
    main()
