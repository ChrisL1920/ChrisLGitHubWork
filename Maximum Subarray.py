class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        current_val = float("-inf")
        max_val = float("-inf")
        for i in nums:
            # if i> current_val + 1:
            # current val = i 
            # else:
            # current_val = current_val + i 
            current_val = max(i, current_val + i)
            max_val = max(max_val, current_val)
        return max_val

        # max_val = float("-inf")
        # for i, n in enumerate(nums):
        #     current_val = 0
        #     for j in range(i,len(nums)):
        #         current_val = current_val + nums[j]
        #         if current_val > max_val:
        #             max_val = current_val
        # return max_val

                
solution =  Solution()
result = solution.maxSubArray([5,4,-1,7,8])
print(result)
    