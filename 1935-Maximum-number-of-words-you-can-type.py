# There is a malfunctioning keyboard where some letter keys do not work. All other keys on the keyboard work properly.
# Given a string text of words separated by a single space (no leading or trailing spaces) and a string brokenLetters of all distinct letter keys that are broken, 
# return the number of words in text you can fully type using this keyboard.
class Solution:
    def canBeTypedWords(self, text: str, brokenLetters: str):
        words = text.split(' ')
        brokenKeys = set(brokenLetters)
        print(brokenKeys)
        num_ok_words = 0 
        for word in words:
            if not any(character in brokenKeys for character in word):
                num_ok_words += 1 
        return num_ok_words

            


solution = Solution()
result = solution.canBeTypedWords(text = "leet code", brokenLetters = "lt")
print(result)