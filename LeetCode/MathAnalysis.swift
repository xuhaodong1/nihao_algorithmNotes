//
//  MathAnalysis.swift
//  LeetCode
//
//  Created by haodong xu on 2022/11/4.
//

import Foundation

/// 数学分析等相关练习题
class MathAnalysis: BaseCode {

    /// 题目链接：[754. 到达终点数字](https://leetcode.cn/problems/reach-a-number/description/)
    /// 参考链接：https://leetcode.cn/problems/reach-a-number/solutions/1947254/fen-lei-tao-lun-xiang-xi-zheng-ming-jian-sqj2/
    func reachNumber(_ target: Int) -> Int {
        let target = abs(target)
        var s = 0, i = 1
        while s < target || (s - target) % 2 == 1 {
            s += i
            i += 1
        }
        return i - 1
    }

    /// 题目链接：[775. 全局倒置与局部倒置](https://leetcode.cn/problems/global-and-local-inversions/description/)
    func isIdealPermutation(_ nums: [Int]) -> Bool {
        return zip((0..<nums.count), nums).allSatisfy { abs($1 - $0) <= 1 }
    }

    /// 题目链接：[891. 子序列宽度之和](https://leetcode.cn/problems/sum-of-subsequence-widths/description/)
    /// 类似题目：
    /// [828. 统计子串中的唯一字符](https://leetcode.cn/problems/count-unique-characters-of-all-substrings-of-a-given-string/)
    /// [907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/)
    /// [2104. 子数组范围和](https://leetcode.cn/problems/sum-of-subarray-ranges/)
    func sumSubseqWidths(_ nums: [Int]) -> Int {
        let nums = nums.sorted(), n = nums.count, MOD = Int(1e9 + 7)
        var ans = 0, pow2s = [Int](repeating: 0, count: n)
        pow2s[0] = 1
        for i in 1..<n {
            pow2s[i] = (pow2s[i - 1] * 2) % MOD
        }
        for (i, num) in nums.enumerated() {
            ans += ((pow2s[i] - pow2s[n - 1 - i]) * num)
        }
        return ((ans % MOD) + MOD) % MOD
    }

    /// 题目链接：[50. Pow(x, n)](https://leetcode.cn/problems/powx-n/description/)
    func myPow(_ x: Double, _ n: Int) -> Double {
        func quickMul(_ x: Double, _ n: Int) -> Double {
            var ans = 1.0, x_contribute = x, n = n
            while n > 0 {
                if n & 1 == 1 { ans *= x_contribute }
                x_contribute *= x_contribute
                n /= 2
            }
            return ans
        }
        return n > 0 ? quickMul(x, n) : 1.0 / quickMul(x, n)
    }

    /// 题目链接：[795. 区间子数组个数](https://leetcode.cn/problems/number-of-subarrays-with-bounded-maximum/description/)
    func numSubarrayBoundedMax(_ nums: [Int], _ left: Int, _ right: Int) -> Int {
        let n = nums.count
        var outIndex = -1, containIndex = -1
        var ans = 0
        for i in 0..<n {
            if nums[i] > right { outIndex = i }
            if nums[i] >= left { containIndex = i }
            ans += containIndex - outIndex
        }
        return ans
    }

    /// 题目链接：[1780. 判断一个数字是否可以表示成三的幂的和](https://leetcode.cn/problems/check-if-number-is-a-sum-of-powers-of-three/description/)
    func checkPowersOfThreeByMath(_ n: Int) -> Bool {
        let maxPower = 15
        var powers3 = [Int](repeating: 1, count: maxPower), n = n
        for i in 1..<maxPower { powers3[i] = powers3[i - 1] * 3 }
        for i in (0..<maxPower).reversed() where n > 0 && powers3[i] <= n { n -= powers3[i] }
        return n == 0
    }

    /// 题目链接：[1780. 判断一个数字是否可以表示成三的幂的和](https://leetcode.cn/problems/check-if-number-is-a-sum-of-powers-of-three/description/)
    func checkPowersOfThreeBySystem(_ n: Int) -> Bool {
        var n = n
        while n != 0 {
            if n % 3 == 2 { return false }
            n /= 3
        }
        return true
    }

    /// 题目链接：[1759. 统计同构子字符串的数目](https://leetcode.cn/problems/count-number-of-homogenous-substrings/description/)
    func countHomogenous(_ s: String) -> Int {
        var ans = 0, pre: Character = s.first!, preCnt = 0
        for c in s {
            if c == pre { preCnt += 1 }
            else { preCnt = 1 }
            ans += preCnt
            pre = c
        }
        return ans % Int(1e9 + 7)
    }

    /// 题目链接：[1250. 检查「好数组」](https://leetcode.cn/problems/check-if-it-is-a-good-array/description/)
    /// 翡蜀定理
    func isGoodArray(_ nums: [Int]) -> Bool {
        func gcd(_ a: Int,_ b: Int) -> Int {
            return b == 0 ? a : gcd(b, a % b)
        }
        return nums.reduce(0) { gcd($1, $0) } == 1
    }

    /// 题目链接：[1238. 循环码排列](https://leetcode.cn/problems/circular-permutation-in-binary-representation/description/)
    func circularPermutation(_ n: Int, _ start: Int) -> [Int] {
        return (0..<(1 << n)).map { ($0 >> 1) ^ $0 ^ start }
    }

    func storeWater(_ bucket: [Int], _ vat: [Int]) -> Int {
        let n = bucket.count
        var res = Int.max, maxK = vat.max()!
        if maxK == 0 { return 0 }
        var k = 1
        while k <= maxK && k < res {
            var t = 0
            for i in 0..<n {
                t += max(0, (vat[i] + k - 1) / k - bucket[i])
            }
            res = min(res, t + k)
            k += 1
        }
        return res
    }

    func minLength(_ s: String) -> Int {
        var chars = [Character](s)
        while chars.count >= 2 {
            let n = chars.count
            var idx = [Int]()
            for i in 0..<n-1 {
                if chars[i] == "A" && chars[i + 1] == "B" {
                    idx.append(i)
                    idx.append(i + 1)
                } else if chars[i] == "C" && chars[i + 1] == "D" {
                    idx.append(i)
                    idx.append(i + 1)
                }
            }
            if idx.count == 0 { break }
            for i in idx.reversed() {
                chars.remove(at: i)
            }
        }
        return chars.count
    }

    func makeSmallestPalindrome(_ s: String) -> String {
        var chars = [Character](s)
        let n = chars.count
        for i in 0..<n/2 {
            if chars[i] != chars[n - i - 1] {
                chars[i] = min(chars[n - i - 1], chars[i])
                chars[n - i - 1] = chars[i]
            }
        }
        return String(chars)
    }

    func punishmentNumber(_ n: Int) -> Int {
        var ans = 0
        var chars = [Character]()
        var find = false
        for i in 1...n {
            let v = i * i
            chars = [Character]("\(v)")
            find = false
            dfs(i: i, curr: 0, sum: 0)
            if find {
                ans += v
            }
        }
        func dfs(i: Int, curr: Int, sum: Int) {
            if find { return }
            if curr >= chars.count {
                find = (sum == i || find)
                return
            }
            var v = 0
            for j in curr..<chars.count {
                v = v * 10 + Int("\(chars[j])")!
                dfs(i: i, curr: j + 1, sum: sum + v)
            }
        }
        return ans
    }

    func sufficientSubset(_ root: TreeNode?, _ limit: Int) -> TreeNode? {
        func dfs(_ node: TreeNode?, _ sum: Int) -> Bool {
            guard let node = node else { return false }
            if node.left == nil && node.right == nil {
                return sum + node.val >= limit
            }
            let left = dfs(node.left, sum + node.val)
            let right = dfs(node.right, sum + node.val)
            if !left {
                node.left = nil
            }
            if !right {
                node.right = nil
            }
            return left || right
        }
        let have = dfs(root, 0)
        return have ? root : nil
    }

    func frogPosition(_ n: Int, _ edges: [[Int]], _ t: Int, _ target: Int) -> Double {
        var g = [[Int]](repeating: [], count: n + 1)
        for edge in edges {
            g[edge[0]].append(edge[1])
            g[edge[1]].append(edge[0])
        }
        var visvited = [Bool](repeating: false, count: n + 1)
        visvited[1] = true
        func dfs(curr: Int, currT: Int) -> Double {
            if currT == t { return curr == target ? 1.0 : 0.0 }
            let cnt = g[curr].filter({ !visvited[$0] }).count
            if cnt == 0 {
                return curr == target ? 1.0 : 0.0
            }
            let res: Double = 1.0 / Double(cnt)
            for next in g[curr] where !visvited[next] {
                visvited[next] = true
                let v = dfs(curr: next, currT: currT + 1)
                if v > 0 {
                    return res * v
                }
                visvited[next] = false
            }
            return 0.0
        }
        return dfs(curr: 1, currT: 0)
    }

    func oddString(_ words: [String]) -> String {
        let n = words.count, m = words[0].count
        let arr = words.map { value($0) }
        if arr[0] == arr[1] {
            for i in 2..<n {
                if arr[i] != arr[0] {
                    return words[i]
                }
            }
        }
        func value(_ str: String) -> String {
            let chars = [Character](str)
            var arr = [Int]()
            for i in 1..<m {
                arr.append(Int(chars[i].asciiValue!) - Int(chars[i - 1].asciiValue!))
            }
            return arr.map { "\($0)" }.joined(separator: "-")
        }
        return arr[0] == arr[2] ? words[1] : words[0]
    }

    func shortestPathBinaryMatrix(_ grid: [[Int]]) -> Int {
        let n = grid.count
        let dirs = [[-1, -1], [0, -1], [-1, 0], [1, 1], [1, 0], [0, 1], [-1, 1], [1, -1]]
        var visited = [[Bool]](repeating: [Bool](repeating: false, count: n), count: n)
        visited[0][0] = true
        var queue = [(0, 0)]
        var length = 1
        if grid[0][0] == 1 || grid[n - 1][n - 1] == 1 { return -1 }
        while !queue.isEmpty {
            let m = queue.count
            for _ in 0..<m {
                let curr = queue.removeFirst()
                if curr == (n - 1, n - 1) { return length }
                for dir in dirs {
                    let nextX = curr.1 + dir[0]
                    let nextY = curr.0 + dir[1]
                    if (nextY >= 0 && nextY < n && nextX >= 0 && nextX < n) && !visited[nextY][nextX] && grid[nextY][nextX] == 0 {
                        visited[nextY][nextX] = true
                        queue.append((nextY, nextX))
                    }
                }
            }
            length += 1
        }
        return -1
    }

    func sampleStats(_ count: [Int]) -> [Double] {
        let totalCnt = count.reduce(0, +)
        var ans = [256.0, 0.0, 0.0, 0.0, 0.0]
        var maxCnt = 0, totalSum = 0.0
        var left = (totalCnt + 1) / 2
        var right = (totalCnt + 2) / 2
        var curr = 0
        for (i, cnt) in count.enumerated() where cnt != 0 {
            ans[0] = min(ans[0], Double(i))
            ans[1] = max(ans[1], Double(i))
            totalSum += Double(cnt * i)
            if maxCnt < cnt {
                maxCnt = cnt
                ans[4] = Double(i)
            }
            if left >= curr && left <= curr + cnt {
                left = i
            }
            if right >= curr && right <= curr + cnt {
                right = i
            }
            curr += cnt
        }
        ans[2] = totalSum / Double(totalCnt)
        ans[3] = (Double(left) + Double(right)) / 2
        return ans
    }

    func kthSmallest(_ mat: [[Int]], _ k: Int) -> Int {
        var a = [0]
        for row in mat {
            var b = [Int](repeating: 0, count: row.count * a.count)
            var i = 0
            for x in a {
                for y in row {
                    b[i] = x + y
                    i += 1
                }
            }
            b.sort()
            if b.count > k {
                b = [Int](b[0..<k])
            }
            a = b
        }
        return a[k - 1]
    }

    func removeTrailingZeros(_ num: String) -> String {
        var chars = [Character](num)
        while chars.count > 1 && chars.last == "0" {
            chars.removeLast()
        }
        return String(chars)
    }

    func differenceOfDistinctValues(_ grid: [[Int]]) -> [[Int]] {
        let n = grid.count, m = grid[0].count
        var ans = [[Int]](repeating: [Int](repeating: 0, count: m), count: n)
        for i in 0..<n {
            for j in 0..<m {
                var set1 = Set<Int>()
                var currI = i - 1, currJ = j - 1
                while currI >= 0 && currJ >= 0 {
                    set1.insert(grid[currI][currJ])
                    currI -= 1
                    currJ -= 1
                }
                var set2 = Set<Int>()
                currI = i + 1; currJ = j + 1
                while currI < n && currJ < m {
                    set2.insert(grid[currI][currJ])
                    currI += 1
                    currJ += 1
                }
                ans[i][j] = abs(set1.count - set2.count)
            }
        }
        return ans
    }

    func minimumCost(_ s: String) -> Int {
        let n = s.count
        let chars = [Character](s)
        var l = n / 2, r = n / 2
        if n & 1 == 0 {
            l -= 1
        }
        func find(l: Int, r: Int, target: Int) -> Int {
            var ans = 0, curr = target
            var l = l, r = r
            while l >= 0 {
                if Int(String(chars[l])) != curr {
                    ans += l + 1
                    while l >= 0 && Int(String(chars[l])) == curr ^ 1 {
                        l -= 1
                    }
                    curr ^= 1
                } else {
                    l -= 1
                }
            }
            curr = target
            while r < n {
                if Int(String(chars[r])) != curr {
                    ans += (n - r)
                    while r < n && Int(String(chars[r])) == curr ^ 1 {
                        r += 1
                    }
                    curr ^= 1
                } else {
                    r += 1
                }
            }
            return ans
        }
        return min(find(l: l, r: r, target: 0), find(l: l, r: r, target: 1))
    }

    func averageValue(_ nums: [Int]) -> Int {
        let arr = nums.filter{ $0 % 6 == 0 }
        return arr.count > 0 ? arr.reduce(0, +) / arr.count : 0
    }

    func delNodes(_ root: TreeNode?, _ to_delete: [Int]) -> [TreeNode?] {
        let set = Set<Int>(to_delete)
        var ans = [TreeNode?]()
        func dfs(curr: TreeNode?, isHead: Bool) {
            guard let curr = curr else { return }
            if !set.contains(curr.val) && isHead {
                ans.append(curr)
            }
            if let left = curr.left {
                if set.contains(left.val) {
                    curr.left = nil
                    dfs(curr: left.left, isHead: true)
                    dfs(curr: left.right, isHead: true)
                } else {
                    dfs(curr: left, isHead: set.contains(curr.val))
                }
            }
            if let right = curr.right {
                if set.contains(right.val) {
                    curr.right = nil
                    dfs(curr: right.left, isHead: true)
                    dfs(curr: right.right, isHead: true)
                } else {
                    dfs(curr: right, isHead: set.contains(curr.val))
                }
            }
        }
        dfs(curr: root, isHead: true)
        return ans
    }

    func applyOperations(_ nums: [Int]) -> [Int] {
        let n = nums.count
        var ans = [Int](), nums = nums
        for i in 0..<n - 1 {
            if nums[i] == nums[i + 1] {
                nums[i] *= 2
                nums[i + 1] = 0
            }
            if nums[i] != 0 {
                ans.append(nums[i])
            }
        }
        ans.append(nums[n - 1])
        ans.append(contentsOf: [Int](repeating: 0, count: n - ans.count))
        return ans
    }

    func mctFromLeafValues(_ arr: [Int]) -> Int {
        let n = arr.count
        var dp = [[Int]](repeating: [Int](repeating: Int.max / 4, count: n), count: n)
        var maxValues = [[Int]](repeating: [Int](repeating: 0, count: n), count: n)
        for j in 0..<n {
            maxValues[j][j] = arr[j]
            dp[j][j] = 0
            for i in (0..<j).reversed() {
                maxValues[i][j] = max(arr[i], maxValues[i + 1][j])
                for k in i..<j {
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + maxValues[i][k] * maxValues[k + 1][j])
                }
            }
        }
        return dp[0][n - 1]
    }

    func maximumTastiness(_ price: [Int], _ k: Int) -> Int {
        let price = price.sorted(), n = price.count
        var l = 0, r = (price[n - 1] - price[0]) / (k - 1)
        while l < r {
            let mid = (l + r + 1) >> 1
            if find(target: mid) >= k {
                l = mid
            } else {
                r = mid - 1
            }
        }
        func find(target: Int) -> Int {
            var pre = price[0], cnt = 1
            for p in price {
                if p >= pre + target {
                    cnt += 1
                    pre = p
                }
            }
            return cnt
        }
        return l
    }

    func vowelStrings(_ words: [String], _ queries: [[Int]]) -> [Int] {
        let vowelChars: Set<Character> = Set<Character>(["a", "e", "i", "o", "u"])
        let n = words.count, m = queries.count
        var preSum = [Int](repeating: 0, count: n + 1)
        var ans = [Int](repeating: 0, count: m)
        for i in 0..<n {
            preSum[i + 1] = preSum[i] + ((vowelChars.contains(words[i].first!) && vowelChars.contains(words[i].last!)) ? 1 : 0)
        }
        for (i, query) in queries.enumerated() {
            ans[i] = preSum[query[1] + 1] - preSum[query[0]]
        }
        return ans
    }

    func distinctAverages(_ nums: [Int]) -> Int {
        let n = nums.count
        var set = Set<Double>(), nums = zip(nums, [Int](repeating: 0, count: n)).map { $0 }
        while true {
            var minV = 101, minIndex = 0
            var maxV = -1, maxIndex = 0
            for (i, item) in nums.enumerated() where item.1 == 0 {
                if minV > item.0 {
                    minV = item.0
                    minIndex = i
                }
                if maxV < item.0 {
                    maxV = item.0
                    maxIndex = i
                }
            }
            if maxV == -1 {
                break
            }
            nums[minIndex].1 = 1
            nums[maxIndex].1 = 1
            set.insert((Double(minV) + Double(maxV)) / 2.0)
        }
        return set.count
    }

    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(distinctAverages([4,1,4,0,3,5]))
    }
}
