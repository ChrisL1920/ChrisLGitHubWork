#testfile
import unittest
import importlib
Parentheses_file = importlib.import_module('20-Valid-Parentheses')

class TestValidParatheses(unittest.TestCase):
    def setUp(self):
        self.solution = Parentheses_file.Solution()

    def test_emptystring(self):
        result = self.solution.isValid("")
        self.assertFalse(result)

    def test_valid1(self):
        result = self.solution.isValid("()")
        self.assertTrue(result)

    def test_valid2(self):
        result = self.solution.isValid("()[]{}")
        self.assertTrue(result)
    
    def test_invalid1(self):
        result = self.solution.isValid("([)]")
        self.assertFalse(result)
    
    def test_invalid2(self):
        result = self.solution.isValid(")")
        self.assertFalse(result)
    