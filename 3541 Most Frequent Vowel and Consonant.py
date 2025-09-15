#test 

class Solution:
    def maxFreqSum(self, s: str) -> int:
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonant = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k','l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']
         # count each character
        freq = {}
        for letter in s:
            if letter not in freq:
                freq[letter] = 0
            freq[letter] += 1
        
        # find max vowel freq
        # instantiate count of consonant
        max_vowel = 0 
        for v in vowels:
            if v in freq and freq[v] > max_vowel:
                max_vowel = freq[v]
            
        # find max consonant freq
        # instantiate count of consonant
        max_con = 0 
        for c in consonant:
            if c in freq and freq[c] > max_con:
                max_con = freq[c]

        return max_con + max_vowel 

        
        

solution =  Solution()
result = solution.maxFreqSum("successes")
print(result)