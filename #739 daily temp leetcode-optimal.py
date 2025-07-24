#739 daily temp leetcode
from typing import List 

class Solution(object):
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # answer = []
        # for current_index,temp in enumerate(temperatures):
        #     future_index = current_index + 1
        #     while future_index < len(temperatures) and temperatures[future_index] < temperatures[current_index]:
        #         future_index = future_index + 1
        #     if future_index >= len(temperatures):
        #         answer.append(0)
        #     else:
        #         next_warm_day  = future_index - current_index
        #         answer.append(next_warm_day)
        # return answer
        pui = []
        result = [0] * len(temperatures)
        for i,temp in enumerate(temperatures):
            while pui:
                x = pui[-1]
                xtemp = temperatures[x]
                if xtemp < temp:
                    diff = i - x 
                    result[x] = diff
                    pui.pop()
                else:
                    break
            pui.append(i)
        return result

solution =  Solution()
result = solution.dailyTemperatures([73,74,75,71,69,72,76,73])
print(result)

