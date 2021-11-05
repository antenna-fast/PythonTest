class Solution:
    visited = set()

    def movingCount(self, m: int, n: int, k: int) -> int:
        pass_num = 0
        def dfs(i, j):
            if not 0 <= i < m or not 0 <= j < n or not calc_num(i, j, k) or (i, j) in self.visited: return 0
            self.visited.add((i, j))
            return 1 + dfs(i + 1, j) + dfs(i, j + 1)

        def calc_num(i, j, k):
            total = 0
            while i:
                total += i % 10
                i = i // 10

            while j:
                total += j % 10
                j = j // 10

            if total > k: return False
            return True

        return dfs(0, 0)


sol = Solution()
print(sol.movingCount(3, 2, 17))
