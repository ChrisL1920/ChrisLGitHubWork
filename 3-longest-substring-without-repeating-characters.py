class Solution(object):
    def lengthOfLongestSubstring(self, word):
        """
        :type s: str
        :rtype: int
        """
        window = []
        max_win_length = 0 

        for letter in word:
            if letter in window:
                letter_index = window.index(letter)
                window = window[letter_index + 1:]

            window.append(letter)
            if len(window) > max_win_length:
                max_win_length = len(window)
        return max_win_length


solution =  Solution()
result = solution.lengthOfLongestSubstring("scene")
print(result)