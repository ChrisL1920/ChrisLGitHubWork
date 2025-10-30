class Solution:
    def romanToInt(self, s: str) -> int:
        num_dict = {'I':1, 'V':5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M':1000}
        value = 0
        s_len = len(s)
        for i, e in enumerate(s):
            if i+ 1 < s_len:
                next = s[i + 1]
            else:
                next = None
            I_value = num_dict[e]
            if i + 1 < s_len:
                next_value = num_dict[next]
                if I_value < next_value:
                    value -= I_value
                else:
                    value += I_value
            else:
                value += I_value

            # if e == 'I' and (next == 'V' or next == 'X'): 
            #     value -= 1
            # elif e == 'X' and (next == 'L' or next == 'C'):
            #     value -= 10
            # elif e == 'C' and (next == 'D' or next == 'M'):
            #     value -= 100
            # else:
            #     value += I_value
                
        return value


solution =  Solution()
result = solution.romanToInt("MCMXCIV")
print(result)