import unittest

class Solution(object):
    def isValid(self, s):
        if len(s) == 0:
            return False
        stack = []
        for symbol in s:
            if symbol in '( { [':
                stack.append(symbol)
            else:
                if len(stack) == 0:
                    return False
                top = stack[-1]
                matches = (top == '(' and symbol == ')') \
                    or (top == '[' and symbol == ']') \
                    or (top == '{' and symbol == '}')
                if matches:
                    stack.pop()
                else:
                    return False
        if len(stack) == 0:
            return True
        else:
            return False
            