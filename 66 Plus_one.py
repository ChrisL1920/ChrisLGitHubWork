class Solution:
    def plusOne(self, digits):
        number = 0 
        for d in digits:
            number = number * 10 + d   # ex d = 1, number = 0 * 10 + 1 = 1, d = 2, number = 1 * 10 + 2 = 12 etc
        number = number + 1 
        num_list = []
        for num in str(number): 
            num_list.append(int(num))
        return num_list

        
        

            
            

solution = Solution()
result = solution.plusOne([1,2,9])
print(result)
