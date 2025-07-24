class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []

        while len(result) < 2*n:
            for i in range(n):
                result.append(nums[i])
            
        return result