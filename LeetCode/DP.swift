//
//  DP.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/3.
//

import Foundation

/// 动态规划相关练习题
class DynamicProgramming: BaseCode {

    /// 题目链接：[646. 最长数对链](https://leetcode.cn/problems/maximum-length-of-pair-chain/)
    /// dp[i] 的含义：以 i 为结尾的最长数对链
    func findLongestChain(_ pairs: [[Int]]) -> Int {
        let n = pairs.count
        let pairs = pairs.sorted { pair1, pair2 in
            return pair1[1] < pair2[1]
        } // 贪心思想, 将其进行[0] / [1] 排序, 保证了符合条件的 pairs[j][1] < pairs[i][0] 的 j 肯定在 i 之前
        var dp = [Int](repeating: 1, count: n)
        var ans = 1
        for i in 0..<n {
            for j in 0..<i where pairs[j][1] < pairs[i][0] {
                dp[i] = max(dp[j] + 1, dp[i])
            }
            ans = max(ans, dp[i])
        }
        return ans
    }

    /// 题目链接：[801. 使序列递增的最小交换次数](https://leetcode.cn/problems/minimum-swaps-to-make-sequences-increasing/)
    func minSwap(_ nums1: [Int], _ nums2: [Int]) -> Int {
        let n = nums1.count
        var dp = [[Int]](repeating: [Int](repeating: n, count: 2), count: n)
        dp[0][0] = 0
        dp[0][1] = 1
        for i in 1..<n {
            if nums1[i] > nums1[i - 1] && nums2[i] > nums2[i - 1] {
                dp[i][0] = dp[i - 1][0]
                dp[i][1] = dp[i - 1][1] + 1
            }
            if nums1[i] > nums2[i - 1] && nums2[i] > nums1[i - 1] {
                dp[i][0] = min(dp[i][0], dp[i - 1][1])
                dp[i][1] = min(dp[i][1], dp[i - 1][0] + 1)
            }
        }
        return min(dp[n - 1][0], dp[n - 1][1])
    }

    /// 题目链接：[940. 不同的子序列 II](https://leetcode.cn/problems/distinct-subsequences-ii/)
    func distinctSubseqII(_ s: String) -> Int {
        let MOD = Int(1e9 + 7)
        let n = s.count, charA = Character("a").asciiValue!, chars = [Character](s)
        var dp = [Int](repeating: 0, count: 26), total = 0
        for i in 0..<n {
            let index = Int(chars[i].asciiValue! - charA)
            let others = total - dp[index]
            dp[index] = 1 + total
            total = ((dp[index] + others) % MOD + MOD) % MOD
        }
        return dp.reduce(0, +) % MOD
    }
    
    /// 题目链接：[902. 最大为 N 的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)
    /// 参考 https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/solution/shu-wei-dp-tong-yong-mo-ban-xiang-xi-zhu-e5dg/
    func atMostNGivenDigitSet(_ digits: [String], _ n: Int) -> Int {
        let digits = digits.map { Character($0) }, s = [Character]("\(n)")
        var dp = [Int](repeating: -1, count: s.count)
        /// i: 当前的位数
        /// isLimit: 当前是否收到了 n 的约束
        /// isNum: i 前面的数位是否填了数字
        func dfs(i: Int, isLimit: Bool, isNum: Bool) -> Int {
            if i == s.count { return isNum ? 1 : 0 }
            if !isLimit && isNum && dp[i] >= 0 { return dp[i] }
            var res = 0
            if !isNum { res = dfs(i: i + 1, isLimit: false, isNum: false) }
            let up = isLimit ? s[i] : "9"
            for digit in digits {
                if digit > up { break }
                res += dfs(i: i + 1, isLimit: isLimit && digit == up, isNum: true)
            }
            if !isLimit && isNum { dp[i] = res }
            return res
        }
        return dfs(i: 0, isLimit: true, isNum: false)
    }
    
    /// 题目链接：[1235. 规划兼职工作](https://leetcode.cn/problems/maximum-profit-in-job-scheduling/)
    func jobScheduling(_ startTime: [Int], _ endTime: [Int], _ profit: [Int]) -> Int {
        let n = startTime.count
        var jobs = [(Int, Int, Int)]()
        for i in 0..<n {
            jobs.append((startTime[i], endTime[i], profit[i]))
        }
        jobs.sort { $0.1 < $1.1 }
        var dp = [Int](repeating: 0, count: n + 1)
        for i in 1...n {
            let j = binarySearch(i - 2, jobs[i - 1].0)
            dp[i] = max(dp[i - 1], jobs[i - 1].2)
            if jobs[j].1 <= jobs[i - 1].0 {
                dp[i] = max(dp[i - 1], dp[j + 1] + jobs[i - 1].2)
            }
        }
        // 返回 endTime <= upper 的最大下标
        func binarySearch(_ right: Int, _ upper: Int) -> Int {
            var left = 0, right = right
            while left < right {
                let mid = (left + right + 1) / 2
                if jobs[mid].1 <= upper {
                    left = mid
                } else {
                    right = mid - 1
                }
            }
            return left
        }
        return dp[n]
    }

    /// 题目链接：[1668. 最大重复子字符串](https://leetcode.cn/problems/maximum-repeating-substring/description/)
    func maxRepeating(_ sequence: String, _ word: String) -> Int {
        let n = sequence.count, m = word.count
        if n < m { return 0 }
        let sChars = [Character](sequence), wChars = [Character](word)
        var dp = [Int](repeating: 0, count: n)
        for i in m-1..<n {
            var vaild = true
            for j in 0..<m where sChars[i - (m - 1) + j] != wChars[j] {
                vaild = false
                break
            }
            if vaild { dp[i] = (i == m - 1 ? 0 : dp[i - m]) + 1 }
        }
        return dp.max()!
    }

    /// 题目链接：[790. 多米诺和托米诺平铺](https://leetcode.cn/problems/domino-and-tromino-tiling/description/)
    func numTilings(_ n: Int) -> Int {
        let MOD = Int(1e9 + 7)
        var dp = [[Int]](repeating: [Int](repeating: 0, count: 4), count: n + 1)
        dp[0][3] = 1
        for i in 1...n {
            let pre = i - 1
            dp[i][0] = dp[pre][3]
            dp[i][1] = (dp[pre][0] + dp[pre][2]) % MOD
            dp[i][2] = (dp[pre][0] + dp[pre][1]) % MOD
            dp[i][3] = dp[pre].reduce(0, +) % MOD
        }
        return dp[n][3]
    }

    /// 题目链接：[799. 香槟塔](https://leetcode.cn/problems/champagne-tower/)
    func champagneTower(_ poured: Int, _ query_row: Int, _ query_glass: Int) -> Double {
        var dp = [[Double]](repeating: [Double](repeating: 0.0, count: query_row + 1), count: query_row + 1)
        dp[0][0] = Double(poured)
        for i in 0..<query_row {
            for j in 0...i where dp[i][j] > 1.0 {
                dp[i + 1][j] += (dp[i][j] - 1) / 2
                dp[i + 1][j + 1] += (dp[i][j] - 1) / 2
            }
        }
        return min(dp[query_row][query_glass], 1.0)
    }

    /// 题目链接：[808. 分汤](https://leetcode.cn/problems/soup-servings/description/)
    func soupServings(_ n: Int) -> Double {
        let n = Int(ceil(Double(n) / 25.0))
        if n >= 179 { return 1.0 }
        var memo = [[Double]](repeating: [Double](repeating: -1, count: n + 1), count: n + 1)
        func dfs(_ a: Int, _ b: Int) -> Double {
            if a <= 0 && b <= 0 { return 0.5 }
            else if a <= 0 { return 1.0 }
            else if b <= 0 { return 0.0 }
            if memo[a][b] == -1 { memo[a][b] = 0.25 * (dfs(a - 4, b) + dfs(a - 3, b - 1) + dfs(a - 2, b - 2) + dfs(a - 1, b - 3)) }
            return memo[a][b]
        }
        return dfs(n, n)
    }

    /// 题目链接：[813. 最大平均值和的分组](https://leetcode.cn/problems/largest-sum-of-averages/description/)
    func largestSumOfAverages(_ nums: [Int], _ k: Int) -> Double {
        let n = nums.count
        var preSum = [Double](repeating: 0, count: n + 1)
        for i in 1...n {
            preSum[i] = preSum[i - 1] + Double(nums[i - 1])
        }
        var dp = [[Double]](repeating: [Double](repeating: 0, count: k + 1), count: n + 1)
        for i in 1...n {
            dp[i][1] = preSum[i] / Double(i)
        }
        for k in 1...k where k >= 2 {
            for i in 1...n where i >= k {
                for j in 0..<i where j >= k - 1 {
                    dp[i][k] = max(dp[i][k], dp[j][k - 1] + (preSum[i] - preSum[j]) / Double(i - j))
                }
            }
        }
        return dp[n][k]
    }

    /// 题目链接：[1687. 从仓库到码头运输箱子](https://leetcode.cn/problems/delivering-boxes-from-storage-to-ports/description/?languageTags=swift)
    func boxDelivering(_ boxes: [[Int]], _ portsCount: Int, _ maxBoxes: Int, _ maxWeight: Int) -> Int {
        let n = boxes.count
        var p = [Int](repeating: 0, count: n + 1) // 记录码头位置
        var w = [Int](repeating: 0, count: n + 1) // 记录箱子重量
        var neg = [Int](repeating: 0, count: n + 2) // 记录从0到i的连续相邻相同码头数量
        var W = [Int](repeating: 0, count: n + 1) // 重量的前缀和
        for i in 1...n {
            p[i] = boxes[i - 1][0]
            w[i] = boxes[i - 1][1]
            if i > 1 {
                neg[i] = neg[i - 1] + (p[i - 1] != p[i] ? 1 : 0)
            }
            W[i] = W[i - 1] + w[i]
        }
        var f = [Int](repeating: 0, count: n + 1), g = [Int](repeating: 0, count: n + 1)
        var opt = [0]
        for i in 1...n {
            while !opt.isEmpty && (i - opt.first! > maxBoxes || W[i] - W[opt.first!] > maxWeight) {
                opt.removeFirst()
            }
            f[i] = g[opt.first!] + neg[i] + 2
            g[i] = f[i] - neg[i + 1]
            while !opt.isEmpty && g[i] < g[opt.last!] {
                opt.removeLast()
            }
            opt.append(i)
        }
        return f[n]
    }

    /// 题目链接：[1691. 堆叠长方体的最大高度](https://leetcode.cn/problems/maximum-height-by-stacking-cuboids/description/)
    func maxHeight(_ cuboids: [[Int]]) -> Int {
        let n = cuboids.count
        var cuboids = cuboids, ans = 0
        var dp = [Int](repeating: 0, count: n)
        for i in 0..<n { cuboids[i].sort() }
        cuboids.sort { a, b in return a[0] != b[0] ? a[0] < b[0] : (a[1] != b[1] ? a[1] < b[1] : a[2] < b[2]) }
        for i in 0..<n { // LIS
            dp[i] = cuboids[i][2] // i 能装下 j
            for j in 0..<i where cuboids[j][1] <= cuboids[i][1] && cuboids[j][2] <= cuboids[i][2] {
                dp[i] = max(dp[i], dp[j] + cuboids[i][2])
            }
            ans = max(ans, dp[i])
        }
        return ans
    }

    /// 题目链接：[1799. N 次操作后的最大分数和](https://leetcode.cn/problems/maximize-score-after-n-operations/description/)
    /// 状态压缩 + 动态规划
    func maxScore(_ nums: [Int]) -> Int {
        let m = nums.count
        var dp = [Int](repeating: 0, count: 1 << m)
        var gcds = [[Int]](repeating: [Int](repeating: 0, count: m), count: m)
        for i in 0..<m {
            for j in i+1..<m {
                gcds[i][j] = gcd(nums[i], nums[j])
            }
        }
        for i in 0..<(1 << m) where i.nonzeroBitCount & 1 == 0 {
            for j in 0..<m where (i >> j) & 1 == 1 {
                for k in j+1..<m where (i >> k) & 1 == 1 {
                    dp[i] = max(dp[i], dp[i ^ (1 << k) ^ (1 << j)] + (i.nonzeroBitCount / 2) * gcds[j][k])
                }
            }
        }
        func gcd(_ a: Int, _ b: Int) -> Int {
            return b == 0 ? a : gcd(b, a % b)
        }
        return dp[(1 << m) - 1]
    }

    /// 题目链接：[2560. 打家劫舍 IV](https://leetcode.cn/problems/house-robber-iv/)
    /// DP + 二分
    func minCapability(_ nums: [Int], _ k: Int) -> Int {
        let n = nums.count
        var l = 0, r = Int(1e9)
        while l < r {
            let mid = (l + r) >> 1
            var dp = [Int](repeating: 0, count: n + 2)
            for (i, num) in nums.enumerated() {
                dp[i + 2] = dp[i + 1]
                if num <= mid {
                    dp[i + 2] = max(dp[i + 2], dp[i] + 1)
                }
            }
            if dp[n + 1] >= k {
                r = mid
            } else {
                l = mid + 1
            }
        }
        return l
    }

    /// 题目链接：[1223. 掷骰子模拟](https://leetcode.cn/problems/dice-roll-simulation/description/)
    /// 先回溯 -> 记忆化搜索 -> 转为DP
    /// 记忆化转DP：
    /// 1.dfs改为f数组
    /// 2.递归改为循环（每一个参数对应一层循环）
    /// 3.递归边界改为f数组的初始值
    func dieSimulator(_ n: Int, _ rollMax: [Int]) -> Int {
        let MOD = Int(1e9 + 7)
        var dp = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: 16), count: 6), count: n)
        for j in 0..<6 {
            for cnt in 0...15 {
                dp[0][j][cnt] = 1
            }
        }
        for i in 1..<n {
            for last in 0..<6 {
                for cnt in 1...rollMax[last] {
                    var res = 0
                    for j in 0..<6 {
                        if j != last { res += dp[i - 1][j][1] }
                        else if rollMax[j] > cnt { res += dp[i - 1][j][cnt + 1] }
                    }
                    dp[i][last][cnt] = res % MOD
                }
            }
        }
        return dp[n - 1].map { $0[1] }.reduce(0, +) % MOD
    }

    /// 题目链接：[1140. 石子游戏 II](https://leetcode.cn/problems/stone-game-ii/)
    func stoneGameII(_ piles: [Int]) -> Int {
        let n = piles.count
        var sufSum = piles
        var memo = [[Int]](repeating: [Int](repeating: -1, count: (n + 1) / 4 + 1), count: n)
        for i in (0..<n).reversed() where i < n - 1 {
            sufSum[i] += sufSum[i + 1]
        }
        func dfs(_ curr: Int, _ m: Int) -> Int {
            if curr + m * 2 >= sufSum.count { return sufSum[curr] }
            if memo[curr][m] != -1 { return memo[curr][m] }
            var res = Int.max
            for x in 1...(2*m) {
                res = min(res, dfs(x + curr, max(m, x)))
            }
            memo[curr][m] = sufSum[curr] - res
            return sufSum[curr] - res
        }
        return dfs(0, 1)
    }

    /// 题目链接：[2572. 无平方子集计数](https://leetcode.cn/problems/count-the-number-of-square-free-subsets/description/)
    func squareFreeSubsets(_ nums: [Int]) -> Int {
        let PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        let MOD = Int(1e9 + 7), MX = 30, N_PRIMES = PRIMES.count, M = 1 << N_PRIMES
        var NSQ_TO_MASK = [Int](repeating: 0, count: MX + 1)
        for i in 2...MX {
            for j in 0..<N_PRIMES where i % PRIMES[j] == 0 {
                if i % (PRIMES[j] * PRIMES[j]) == 0 {
                    NSQ_TO_MASK[i] = -1
                    break
                }
                NSQ_TO_MASK[i] |= (1 << j)
            }
        }
        var dp = [Int](repeating: 0, count: M)
        dp[0] = 1
        for x in nums {
            let mask = NSQ_TO_MASK[x]
            if mask >= 0 {
                for j in (mask...M-1).reversed() where (j | mask) == j {
                    dp[j] = (dp[j] + dp[j ^ mask]) % MOD
                }
            }
        }
        return (dp.reduce(0, +) - 1) % MOD
    }

    func printBin(_ num: Double) -> String {
        var ans = "0.", num = num, cnt = 0
        while ans.count <= 32 && num != 0 {
            num *= 2
            let digit = Int(num)
            ans += "\(digit)"
            num -= Double(digit)
            cnt += 1
        }
        return ans.count <= 32 ? ans : "ERROR"
    }

    func passThePillow(_ n: Int, _ time: Int) -> Int {
        let t = time % (n - 1)
        return time / (n - 1) & 1 == 0 ? 1 + t : n - t
    }

    func kthLargestLevelSum(_ root: TreeNode?, _ k: Int) -> Int {
        guard let root = root else { return -1 }
        var queue: [TreeNode] = [root]
        var arr = [Int]()
        while queue.count > 0 {
            var sum = 0
            var temp = [TreeNode]()
            queue.forEach { curr in
                sum += curr.val
                if let left = curr.left { temp.append(left) }
                if let right = curr.right { temp.append(right) }
            }
            queue = temp
            arr.append(sum)
        }
        arr.sort(by: >)
        return arr.count >= k ? arr[k - 1] : -1
    }

    func findValidSplit(_ nums: [Int]) -> Int {
        let n = nums.count
        var left = [Int: Int]()
        var right = [Int](repeating: -1, count: n)
        func updateArr(p: Int, i: Int) {
            if left.keys.contains(p) { right[left[p]!] = i }
            else { left[p] = i }
        }
        for (i, x) in nums.enumerated() {
            var x = x, d = 2
            while d * d <= x {
                if x % d == 0 {
                    updateArr(p: d, i: i)
                    x /= d
                }
                while x % d == 0 { x /= d }
                d += 1
            }
            if x > 1 { updateArr(p: x, i: i) }
        }
        var maxR = 0
        for (l, r) in right.enumerated() {
            if l > maxR { return maxR }
            maxR = max(maxR, r)
        }
        return -1
    }

    func waysToReachTarget(_ target: Int, _ types: [[Int]]) -> Int {
        let mod = Int(1e9 + 7)
        var dp = [Int](repeating: 0, count: target + 1)
        dp[0] = 1
        for type in types {
            let count = type[0], marks = type[1]
            for j in (1...target).reversed() {
                for k in 1...count where k <= j / marks {
                    dp[j] = (dp[j] + dp[j - k * marks]) % mod
                }
            }
        }
        return dp[target]
    }

    func minOperationsMaxProfit(_ customers: [Int], _ boardingCost: Int, _ runningCost: Int) -> Int {
        let n = customers.count
        var ans = -1, maxCost = 0, wait = 0, curr = 0, i = 0
        while wait > 0 || i < n {
            wait += i >= n ? 0 : customers[i]
            let up = min(4, wait)
            wait -= up
            curr += boardingCost * up - runningCost
            i += 1
            if curr > maxCost {
                maxCost = curr
                ans = i
            }
        }
        return ans
    }

    func vowelStrings(_ words: [String], _ left: Int, _ right: Int) -> Int {
        let yuanyin: [Character] = ["a", "e", "i", "o", "u"]
        return words[left...right].reduce(0) { return $0 + (yuanyin.contains($1.first!) && yuanyin.contains($1.last!) ? 1 : 0) }
    }

    func maxScore1(_ nums: [Int]) -> Int {
        let nums = nums.sorted(by: >)
        var cnt = 0, sum = 0
        for num in nums {
            sum += num
            if sum <= 0 { break }
            cnt += 1
        }
        return cnt
    }

    func beautifulSubarrays(_ nums: [Int]) -> Int {
        let n = nums.count
        var s = [Int](repeating: 0, count: n + 1)
        for i in 0..<n { s[i + 1] = s[i] ^ nums[i] }
        var first = [Int: Int]()
        var ans = 0
        for i in 0...n {
            ans += first[s[i], default: 0]
            first[s[i], default: 0] += 1
        }
        return ans
    }

    func findMinimumTime(_ tasks: [[Int]]) -> Int {
        let tasks = tasks.sorted { $0[1] < $1[1] }
        var ans = 0
        var run = [Bool](repeating: false, count: tasks[tasks.count - 1][1] + 1)
        for task in tasks {
            let start = task[0], end = task[1]
            var d = task[2]
            for i in start...end where run[i] {
                d -= 1
            }
            for i in (1...end).reversed() where d > 0 && !run[i] {
                run[i] = true
                d -= 1
                ans += 1
            }
        }
        return ans
    }

    func countSubgraphsForEachDiameter(_ n: Int, _ edges: [[Int]]) -> [Int] {
        var g = [[Int]](repeating: [], count: n)
        for edge in edges {
            let x = edge[0] - 1, y = edge[1] - 1
            g[x].append(y)
            g[y].append(x)
        }
        var ans = [Int](repeating: 0, count: n - 1)
        var inSet = [Bool](repeating: false, count: n)
        var vis = [Bool]()
        var diameter = 0
        f(i: 0)
        func f(i: Int) {
            if i == n {
                for v in 0..<n {
                    if inSet[v] {
                        vis = [Bool](repeating: false, count: n)
                        diameter = 0
                        dfs(v)
                        break
                    }
                }
                if diameter > 0 && vis == inSet {
                    ans[diameter - 1] += 1
                }
                return
            }
            f(i: i + 1)

            inSet[i] = true
            f(i: i + 1)
            inSet[i] = false
        }
        func dfs(_ x: Int) -> Int {
            vis[x] = true
            var maxLen = 0
            for y in g[x] {
                if !vis[y] && inSet[y] {
                    let ml = dfs(y) + 1
                    diameter = max(diameter, maxLen + ml)
                    maxLen = max(maxLen, ml)
                }
            }
            return maxLen
        }
        return ans
    }

    func countSubarrays(_ nums: [Int], _ k: Int) -> Int {
        let n = nums.count
        let pos = nums.firstIndex(of: k)!
        var x = 0, map: [Int: Int] = [0: 1]
        for i in (0..<pos).reversed() {
            x += nums[i] < k ? 1 : -1
            map[x, default: 0] += 1
        }
        var ans = map[0, default: 0] + map[-1, default: 0]
        x = 0
        for i in pos+1..<n {
            x += nums[i] > k ? 1 : -1
            ans += map[x, default: 0] + map[x - 1, default: 0]
        }
        return ans
    }

    func answerQueries(_ nums: [Int], _ queries: [Int]) -> [Int] {
        let n = nums.count, m = queries.count
        var s = [Int](repeating: 0, count: n + 1)
        for (i, num) in nums.sorted().enumerated() {
            s[i + 1] = s[i] + num
        }
        var ans = [Int](repeating: 0, count: m)
        for (i, query) in queries.enumerated() {
            var l = 0, r = n
            while l < r {
                let mid = (l + r + 1) >> 1
                if query < s[mid] {
                    r = mid - 1
                } else {
                    l = mid
                }
            }
            ans[i] = l
        }
        return ans
    }

    func evenOddBit(_ n: Int) -> [Int] {
        var n = n
        var ans = [0, 0]
        var curr = 0
        while n > 0 {
            ans[curr] += n & 1
            curr ^= 1
            n >>= 1
        }
        return ans
    }

    func checkValidGrid(_ grid: [[Int]]) -> Bool {
        let n = grid.count
        var map = [Int: (Int, Int)]()
        for i in 0..<n {
            for j in 0..<n {
                if map.keys.contains(grid[i][j]) {
                    return false
                } else {
                    map[grid[i][j]] = (i, j)
                }
            }
        }
        var curr = (0, 0)
        for i in 1...(n*n-1) {
            if let (x, y) = map[i] {
                if abs((x - curr.0)) + abs(y - curr.1) != 3 {
                    return false
                }
                curr = (x, y)
            } else {
                return false
            }
        }
        return true
    }

    func beautifulSubsets(_ nums: [Int], _ k: Int) -> Int {
        let n = nums.count
        var ans = -1
        var set = Set<Int>()
        func dfs(i: Int) {
            if i == n { ans += 1; return }
            dfs(i: i + 1)
            if !set.contains(nums[i] - k) && !set.contains(nums[i] + k) {
                set.insert(nums[i])
                dfs(i: i + 1)
                set.remove(nums[i])
            }
        }
        dfs(i: 0)
        return ans
    }

    func findSmallestInteger(_ nums: [Int], _ value: Int) -> Int {
        let n = nums.count
        var map = [Int: Int]()
        for num in nums {
            map[(num % value + value) % value, default: 0] += 1
        }
        for i in 0..<n {
            if let cnt = map[i % value], cnt > 0 {
                map[i % value]! -= 1
            } else {
                return i
            }
        }
        return n
    }

    func numDupDigitsAtMostN(_ n: Int) -> Int {
        let s = "\(n)".map{ $0 }
        let m = s.count
        var memo = [[Int]](repeating: [Int](repeating: -1, count: 1 << 10), count: m)
        func f(_ i: Int, _ mask: Int, _ isLimit: Bool, _ isNum: Bool) -> Int {
            if i == m { return isNum ? 1 : 0 }
            if !isLimit && isNum && memo[i][mask] != -1 { return memo[i][mask] }
            var ans = 0
            if !isNum {
                ans += f(i + 1, mask, false, false)
            }
            let up = isLimit ? Int(s[i].asciiValue!) - Int(Character("a").asciiValue!) : 9
            var d = isNum ? 0 : 1
            while d <= up {
                if (mask >> d) & 1 == 0 {
                    ans += f(i + 1, mask | (1 << d), isLimit && d == up, true)
                }
                d += 1
            }
            if !isLimit && isNum {
                memo[i][mask] = ans
            }
            return ans
        }
        return n - f(0, 0, true, false)
    }

    func bestTeamScore(_ scores: [Int], _ ages: [Int]) -> Int {
        let arr = zip(scores, ages).sorted { item1, item2 in
            return item1.1 == item2.1 ? item1.0 < item2.0 : item1.1 < item2.1
        }
        let n = arr.count
        var ans = 0
        var dp = [Int](repeating: 0, count: n)
        for i in 0..<n {
            dp[i] = arr[i].0
            for j in 0..<i {
                if arr[j].0 <= arr[i].0 {
                    dp[i] = max(dp[i], dp[j] + arr[i].0)
                }
            }
            ans = max(ans, dp[i])
        }
        return ans
    }

    func primeSubOperation(_ nums: [Int]) -> Bool {
        let n = nums.count
        var primes = [Int]()
        for num in 2...1000 {
            var i = 2, isPrime = true
            while i + i <= num {
                if num % i == 0 {
                    isPrime = false
                    break
                }
                i += 1
            }
            if isPrime { primes.append(num) }
        }
        var nums = nums
        for i in (0..<n-1).reversed() {
            if nums[i] >= nums[i + 1] {
                let diff = nums[i] - nums[i + 1]
                for prime in primes {
                    if prime >= nums[i] { break }
                    if prime > diff { nums[i] -= prime; break }
                }
                if nums[i] >= nums[i + 1] {
                    return false
                }
            }
        }
        return true
    }

    func minOperations(_ nums: [Int], _ queries: [Int]) -> [Int] {
        let n = nums.count, m = queries.count, nums = nums.sorted()
        var preSum = [Int](repeating: 0, count: n + 1)
        for i in 0..<n {
            preSum[i + 1] = preSum[i] + nums[i]
        }
        var ans = [Int](repeating: 0, count: m)
        for (i, query) in queries.enumerated() {
            var l = -1, r = n - 1
            while l < r {
                let mid = (r + l + 1) >> 1
                // 最大的<=
                if nums[mid] <= query {
                    l = mid
                } else {
                    r = mid - 1
                }
            }
            ans[i] = query * (l + 1) - preSum[l + 1] + preSum[n] - preSum[l + 1] - query * (n - l - 1)
        }
        return ans
    }

    func shortestCommonSupersequence(_ str1: String, _ str2: String) -> String {
        let str1 = [Character](str1), str2 = [Character](str2)
        let n = str1.count, m = str2.count
        var memo = [[Int]](repeating: [Int](repeating: -1, count: m), count: n)
        func dfs(_ i: Int, _ j: Int) -> Int {
            if i < 0 { return j + 1 }
            if j < 0 { return i + 1 }
            if memo[i][j] != -1 { return memo[i][j] }
            if str1[i] == str2[j] {
                memo[i][j] = dfs(i - 1, j - 1) + 1
            } else {
                memo[i][j] = min(dfs(i - 1, j), dfs(i, j - 1)) + 1
            }
            return memo[i][j]
        }
        func makeAns(_ i: Int, _ j: Int) -> [Character] {
            if i < 0 && j < 0 { return [] }
            if i < 0 { return [Character](str2[0...j]) }
            if j < 0 { return [Character](str1[0...i]) }
            var ans: [Character] = []
            if str1[i] == str2[j] {
                ans = makeAns(i - 1, j - 1)
                ans.append(str1[i])
            } else if dfs(i, j) == dfs(i - 1, j) + 1 {
                ans = makeAns(i - 1, j)
                ans.append(str1[i])
            } else {
                ans = makeAns(i, j - 1)
                ans.append(str2[j])
            }
            return ans
        }
        return String(makeAns(n - 1, m - 1))
    }

    func maskPII(_ s: String) -> String {
        let n = s.count
        var chars = [Character](s)
        var ans = ""
        if chars.contains("@") {
            for i in 0..<n where chars[i].isLetter && chars[i].isUppercase {
                chars[i] = Character(chars[i].lowercased())
            }
            ans += "\(chars[0])"
            ans += "*****\(chars[chars.firstIndex(of: "@")! - 1])@"
            ans += "\(String(chars[chars.firstIndex(of: "@")!+1...n-1]))"
        } else {
            chars = chars.filter({ $0 != "+" && $0 != "-" && $0 != "(" && $0 != ")" && $0 != " " })
            let curr = chars.count - 1
            ans = "-\(String(chars[curr-3...curr]))"
            ans = "***-***" + ans
            if chars.count > 10 {
                ans = "+\(String([Character](repeating: "*", count: curr-9)))-" + ans
            }
        }
        return ans
    }

    func findTheLongestBalancedSubstring(_ s: String) -> Int {
        let chars = [Character](s), n = chars.count
        var ans = 0
        for i in 0..<n {
            for j in i+1..<n {
                if (j - i + 1) & 1 == 0 {
                    var has = true
                    for k in i..<i+(j - i + 1)/2 {
                        if chars[k] != "0" {
                            has = false
                            break
                        }
                    }
                    for l in i+(j - i + 1)/2...j {
                        if chars[l] != "1" {
                            has = false
                            break
                        }
                    }
                    if has { ans = max(j - i + 1, ans) }
                }
            }
        }
        return ans
    }

    func findMatrix(_ nums: [Int]) -> [[Int]] {
        var map = [Int: Int]()
        for num in nums {
            map[num, default: 0] += 1
        }
        var ans = [[Int]]()
        while !map.isEmpty {
            var curr = [Int]()
            for item in map {
                curr.append(item.key)
                map[item.key]! -= 1
                if map[item.key]! == 0 {
                    map.removeValue(forKey: item.key)
                }
            }
            ans.append(curr)
        }
        return ans
    }

    func miceAndCheese(_ reward1: [Int], _ reward2: [Int], _ k: Int) -> Int {
        let n = reward1.count
        let arr = zip(reward1, reward2).sorted { item1, item2 in
            return abs(item1.0 - item1.1) > abs(item2.0 - item2.1)
        }
        var ans = 0, k = k
        for (i, item) in arr.enumerated() {
            if k == 0 {
                ans += item.1
            } else {
                if item.0 >= item.1 { // reward1 < reward2 选
                    ans += item.0
                    k -= 1
                } else if n - i == k { // 只能选 1
                    ans += item.0
                    k -= 1
                } else {
                    ans += item.1
                }
            }
        }
        return ans
    }

//    func minReverseOperations(_ n: Int, _ p: Int, _ banned: [Int], _ k: Int) -> [Int] {
//        banned.swapAt(<#T##Self.Index#>, <#T##Self.Index#>)
//    }

    func prevPermOpt1(_ arr: [Int]) -> [Int] {
        var arr = arr
        for i in (0..<arr.count-1).reversed() where arr[i] > arr[i + 1] {
            var k = i + 1
            for j in i+1..<arr.count where arr[i] > arr[j] && (arr[i] - arr[j] < arr[i] - arr[k]) {
                k = j
            }
            arr.swapAt(k, i)
            break
        }
        return arr
    }

    func baseNeg2(_ n: Int) -> String {
        if n == 0 { return "0" }
        var curr = 0, n = n
        var i = 0
        while n != 0 {
            if n & 1 != 0 {
                if i & 1 == 1 {
                    curr += Int(pow(2.0, Double(i)))
                    n += Int(pow(2.0, Double(1)))
                } else {
                    curr += Int(pow(2.0, Double(i)))
                }
            }
            n >>= 1
            i += 1
        }
        var ans = ""
        while curr > 0 {
            ans = "\(curr & 1)" + ans
            curr >>= 1
        }
        return ans
    }

    func numMovesStonesII(_ stones: [Int]) -> [Int] {
        let stones = stones.sorted(), n = stones.count
        /// 计算空位
        let e1 = stones[n - 2] - stones[0] - n + 2
        let e2 = stones[n - 1] - stones[1] - n + 2
        let maxMove = max(e1, e2)
        if e1 == 0 || e2 == 0 {
            return [min(2, maxMove), maxMove]
        }
        var maxCnt = 0, l = 0
        for r in 0..<n {
            while stones[r] - stones[l] + 1 > n {
                l += 1
            }
            maxCnt = max(maxCnt, r - l + 1)
        }
        return [n - maxCnt, maxMove]
    }

    func checkDistances(_ s: String, _ distance: [Int]) -> Bool {
        var map = [Character: Int]()
        for (i, char) in s.enumerated() {
            if let j = map[char] {
                if distance[Int(char.asciiValue! - Character("a").asciiValue!)] != j - i - 1 {
                    return false
                }
            } else {
                map[char] = i
            }
        }
        return true
    }

    func diagonalPrime(_ nums: [[Int]]) -> Int {
        let n = nums.count
        var ans = 0
        for i in 0..<n {
            if isPrime(num: nums[i][i]) {
                ans = max(ans, nums[i][i])
            }
            if isPrime(num: nums[i][n - i - 1]) {
                ans = max(ans, nums[i][n - i - 1])
            }
        }
        func isPrime(num: Int) -> Bool {
            if num < 2 { return false }
            var i = 2, isPrime = true
            while i * i <= num {
                if num % i == 0 {
                    isPrime = false
                    break
                }
                i += 1
            }
            return isPrime
        }
        return ans
    }

    func distance(_ nums: [Int]) -> [Int] {
        let n = nums.count
        var leftMap = [Int: [Int]]()
        var ans = [Int](repeating: 0, count: n)
        for (i, num) in nums.enumerated() {
            leftMap[num, default: []].append(i)
        }
        for item in leftMap {
            let cnt = item.value.count
            var sum = 0
            for i in 0..<cnt {
                sum += item.value[i] - item.value[0]
            }
            ans[0] = sum
            for i in 0..<cnt {
                sum -= (i > 0 ? item.value[i] - item.value[i - 1] : 0) * (cnt - i)
                ans[item.value[i]] = sum
            }
            for i in (0..<cnt).reversed() {
                sum += item.value[cnt - 1] - item.value[i]
            }
            for i in (0..<cnt).reversed() {
                sum -= (i < cnt - 1 ? item.value[i + 1] - item.value[i] : 0) * (i + 1)
                ans[item.value[i]] += sum
            }
        }
        return ans
    }

    func minimizeMax(_ nums: [Int], _ p: Int) -> Int {
        let n = nums.count
        let nums = nums.sorted()
        var l = 0, r = nums.last!
        while l < r {
            let mid = (l + r) >> 1
            var cnt = 0
            var i = 0
            while i < n - 1 {
                if nums[i + 1] - nums[i] <= mid {
                    cnt += 1
                    i += 1
                }
                i += 1
            }
            if cnt >= p {
                r = mid
            } else {
                l = mid + 1
            }
        }
        return r
    }

    func nextLargerNodes(_ head: ListNode?) -> [Int] {
        var curr = head, n = 0
        while curr != nil {
            n += 1
            curr = curr?.next
        }
        var queue = [(Int, Int)]()
        var ans = [Int](repeating: 0, count: n)
        var i = 0
        curr = head
        while curr != nil {
            while !queue.isEmpty && curr!.val > queue.last!.1 {
                ans[queue.removeLast().0] = curr!.val
            }
            queue.append((i, curr!.val))
            i += 1
            curr = curr?.next
        }
        return ans
    }

    func isRobotBounded(_ instructions: String) -> Bool {
        let dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        var dir = 0, position = [0, 0]
        for instruction in instructions {
            if instruction == "G" {
                position[0] += dirs[dir][0]
                position[1] += dirs[dir][1]
            } else if instruction == "L" {
                dir = (dir + 3) % 4
            } else if instruction == "R" {
                dir = (dir + 1) % 4
            }
        }
        return dir != 0 || position == [0, 0]
    }

    func longestDecomposition(_ text: String) -> Int {
        if text.isEmpty || text.count == 1 { return text.count }
        let n = text.count
        for i in 1...n/2 {
            if text[text.startIndex..<text.index(text.startIndex, offsetBy: i)] == text[text.index(text.endIndex, offsetBy: -i)..<text.endIndex] {
                return longestDecomposition(String(text[text.index(text.startIndex, offsetBy: i)..<text.index(text.endIndex, offsetBy: -i)])) + 2
            }
        }
        return 1
    }

    func minimizeArrayValue(_ nums: [Int]) -> Int {
        var l = 0, r = nums.max()!
        while l < r {
            let mid = (l + r) >> 1
            if check(mid) {
                r = mid
            } else {
                l = mid + 1
            }
        }
        func check(_ limit: Int) -> Bool {
            let n = nums.count
            var last = nums[n - 1]
            for i in (1..<n).reversed() {
                if last > limit {
                    last = nums[i - 1] + last - limit
                } else {
                    last = nums[i - 1]
                }
            }
            return last <= limit
        }
        return l
    }

    func minimizeSet(_ divisor1: Int, _ divisor2: Int, _ uniqueCnt1: Int, _ uniqueCnt2: Int) -> Int {
        var l = 2, r = 100
        let lcmValue = lcm(a: divisor1, b: divisor2)
        while l < r {
            let mid = (l + r) >> 1
            if check(mid) {
                r = mid
            } else {
                l = mid + 1
            }
        }
        func gcd(a: Int, b: Int) -> Int {
            return b == 0 ? a : gcd(a: b, b: a % b)
        }
        func lcm(a: Int, b: Int) -> Int {
            return a * b / gcd(a: a, b: b)
        }
        func check(_ limit: Int) -> Bool {
            return limit - limit / divisor1 >= uniqueCnt1 && limit - limit / divisor2 >= uniqueCnt2 && limit - limit / lcmValue >= uniqueCnt1 + uniqueCnt2
        }
        return l
    }

    func mostFrequentEven(_ nums: [Int]) -> Int {
        var map = [Int: Int]()
        for num in nums where num & 1 == 0 {
            map[num, default: 0] += 1
        }
        return map.max { item1, item2 in
            return item1.value == item2.value ? item1.key < item2.key : item1.value > item2.value
        }?.value ?? -1
    }

    func camelMatch(_ queries: [String], _ pattern: String) -> [Bool] {
        let n = queries.count, m = pattern.count, pattern = [Character](pattern)
        var ans = [Bool](repeating: false, count: n)
        for i in 0..<n {
            ans[i] = check([Character](queries[i]))
        }
        func check(_ query: [Character]) -> Bool {
            let k = query.count
            var i = 0, j = 0
            while i < k || j < m {
                if k - i < m - j {
                    return false
                } else if j < m && query[i] == pattern[j] {
                    j += 1
                } else if query[i].isUppercase {
                    return false
                }
                i += 1
            }
            return j == m
        }
        return ans
    }

    func rowAndMaximumOnes(_ mat: [[Int]]) -> [Int] {
        var ans = 0, maxSum = 0
        for i in 0..<mat.count {
            var curr = 0
            for j in 0..<mat[0].count {
                if mat[i][j] == 1 {
                    curr += 1
                }
            }
            if curr > maxSum {
                ans = i
                maxSum = curr
            }
        }
        return [ans, maxSum]
    }

    func maxDivScore(_ nums: [Int], _ divisors: [Int]) -> Int {
        var ans = Int.max, maxSum = 0
        for divisor in divisors {
            var curr = 0
            for num in nums {
                if num % divisor == 0 {
                    curr += 1
                }
            }
            if divisor == 7 {
                print(curr)
            }
            if divisor == 25 {
                print(curr)
            }
            if maxSum == curr {
                ans = min(ans, divisor)
            } else if maxSum < curr {
                ans = divisor
                maxSum = curr
            }
        }
        return ans
    }

    func addMinimum(_ word: String) -> Int {
        let chars = [Character](word), n = chars.count
        var ans = 0
        var i = 0
        while i < n {
            if i == n - 1 {
                ans += 2
                i += 1
            } else if chars[i].asciiValue! >= chars[i + 1].asciiValue! {
                ans += 2
                i += 1
            } else if chars[i].asciiValue! == chars[i + 1].asciiValue! - 1 {
                if chars[i] == "a" && i < n - 2 && chars[i + 2] == "c" {
                    i += 3
                } else if chars[i] == "a" && i < n - 2 && chars[i + 2] != "c" {
                    ans += 1
                    i += 2
                } else if chars[i] == "a" && i == n - 2 {
                    ans += 1
                    i += 2
                } else if chars[i] == "b" {
                    ans += 1
                    i += 2
                }
            } else if chars[i].asciiValue! == chars[i + 1].asciiValue! - 2 {
                ans += 1
                i += 2
            }
        }
        return ans
    }

    func minimumTotalPrice(_ n: Int, _ edges: [[Int]], _ price: [Int], _ trips: [[Int]]) -> Int {
        var arr = [[Int]](repeating: [], count: n)
        for edge in edges {
            arr[edge[0]].append(edge[1])
            arr[edge[1]].append(edge[0])
        }
        var visvit = Set<Int>()
        func dfs(_ i: Int, _ target: Int, _ path: [Int]) -> [Int] {
            if i == target {
                return path
            }
            var path = path
            for j in arr[i] where !visvit.contains(j) {
                path.append(j)
                visvit.insert(j)
                let currPath = dfs(j, target, path)
                path.removeLast()
                if currPath.count > 0 {
                    return currPath
                }
            }
            return []
        }
        var map = [Int: Int]()
        for trip in trips {
            visvit = [trip[0]]
            let arr = dfs(trip[0], trip[1], [trip[0]])
            for item in arr {
                map[item, default: 0] += 1
            }
        }
        func dfs(_ i: Int) -> [Int] {
            var notHalf = map[i, default: 0] * price[i]
            var half = notHalf / 2
            for j in arr[i] where !visvit.contains(j) {
                visvit.insert(j)
                let res = dfs(j)
                notHalf += min(res[0], res[1])
                half += res[0]
            }
            return [notHalf, half]
        }
        visvit = [0]
        return dfs(0).min()!
    }

    func maxAncestorDiff(_ root: TreeNode?) -> Int {
        guard let root = root else { return 0 }
        var ans = 0
        func dfs(_ curr: TreeNode) -> (Int, Int) {
            let val = curr.val
            var leftVal = (val, val)
            var rightVal = (val, val)
            if let left = curr.left {
                leftVal = dfs(left)
            }
            if let right = curr.right {
                rightVal = dfs(right)
            }
            ans = max(absInt(val: val - leftVal.0), absInt(val: val - leftVal.1), absInt(val: val - rightVal.0), absInt(val: val - rightVal.1), ans)
            return (min(val, leftVal.0, rightVal.0), max(val, leftVal.1, rightVal.1))
        }
        func absInt(val: Int) -> Int {
            return val >= 0 ? val : -val
        }
        dfs(root)
        return ans
    }

    func gardenNoAdj(_ n: Int, _ paths: [[Int]]) -> [Int] {
        var g = [[Int]](repeating: [], count: n)
        for path in paths {
            g[path[0] - 1].append(path[1] - 1)
            g[path[1] - 1].append(path[0] - 1)
        }
        var colors = [Int](repeating: 0, count: n)
        for i in 0..<n {
            var used = [Bool](repeating: false, count: 5)
            for j in g[i] {
                used[colors[j]] = true
            }
            for c in 1...4 where !used[c] {
                colors[i] = c
                break
            }
        }
        return colors
    }

    func maxSumAfterPartitioning(_ arr: [Int], _ k: Int) -> Int {
        let n = arr.count
        var dp = [Int](repeating: 0, count: n + 1)
        for i in 0..<n {
            var mx = 0
            for j in (max(0, i - k + 1)...i).reversed() {
                mx = max(mx, arr[j])
                dp[i + 1] = max(dp[i + 1], dp[j] + (i - j + 1) * mx)
            }
        }
        return dp[n]
    }

    func validPartition(_ nums: [Int]) -> Bool {
        let n = nums.count
        var dp = [Bool](repeating: false, count: n + 1)
        dp[0] = true
        for i in 0..<n {
            if i >= 1 && nums[i] == nums[i - 1] {
                dp[i + 1] = dp[i + 1] || dp[i - 1]
            }
            if i >= 2 && nums[i] == nums[i - 1] && nums[i] == nums[i - 2] {
                dp[i + 1] = dp[i + 1] || dp[i - 2]
            }
            if i >= 2 && nums[i] == nums[i - 1] + 1 && nums[i] == nums[i - 2] + 2 {
                dp[i + 1] = dp[i + 1] || dp[i - 2]
            }
        }
        return dp[n]
    }

    func makeArrayIncreasing(_ arr1: [Int], _ arr2: [Int]) -> Int {
        let arr2 = arr2.sorted()
        let n = arr1.count
        var memo = [[Int: Int]](repeating: [Int: Int](), count: n)
        let ans = dfs(n - 1, Int.max)
        func dfs(_ i: Int, _ pre: Int) -> Int {
            if i < 0 { return 0 }
            if memo[i].keys.contains(pre) {
                return memo[i][pre]!
            }
            var res = arr1[i] < pre ? dfs(i - 1, arr1[i]) : Int.max / 2
            let k = lowerBound(pre)
            if k >= 0 {
                res = min(res, dfs(i - 1, arr2[k]) + 1)
            }
            memo[i][pre] = res
            return res
        }
        func lowerBound(_ target: Int) -> Int {
            var l = -1, r = arr2.count - 1
            while l < r {
                let mid = (l + r + 1) >> 1
                if arr2[mid] < target {
                    l = mid
                } else {
                    r = mid - 1
                }
            }
            return l
        }
        return ans < Int.max / 2 ? ans : -1
    }

    func lastSubstring(_ s: String) -> String {
        let chars = [Character](s), n = chars.count
        var i = 0, j = 1, k = 0
        while j + k < n {
            if chars[i + k] == chars[j + k] {
                k += 1
            } else if chars[i + k] > chars[j + k] {
                j += k + 1
                k = 0
            } else {
                i += k + 1
                k = 0
                if i >= j {
                    j = i + 1
                }
            }
        }
        return String(chars[i..<n])
    }

    func sortPeople(_ names: [String], _ heights: [Int]) -> [String] {
        let n = names.count
        let arr = (0..<n).sorted(by: { heights[$0] > heights[$1] })
        var ans = [String]()
        for i in arr {
            ans.append(names[i])
        }
        return ans
    }

    func maxSumTwoNoOverlap(_ nums: [Int], _ firstLen: Int, _ secondLen: Int) -> Int {
        let n = nums.count
        var dp = [[Int]](repeating: [0, 0], count: n + 1)
        var sumF = 0, sumS = 0
        var ans = 0
        for i in 0..<n {
            sumF += nums[i]
            sumS += nums[i]
            if i >= firstLen {
                sumF -= nums[i - firstLen]
            }
            if i >= secondLen {
                sumS -= nums[i - secondLen]
            }
            dp[i + 1][0] = max(dp[i][0], sumF)
            dp[i + 1][1] = max(dp[i][1], sumS)
            ans = max(ans, sumF + dp[i - min(firstLen - 1, i)][1], sumS + dp[i - min(secondLen - 1, i)][0])
        }
        return ans
    }

    func longestStrChain(_ words: [String]) -> Int {
        let n = words.count, words = words.sorted { $0.count < $1.count }.map { [Character]($0) }
        var dp = [Int](repeating: 1, count: n + 1)
        var ans = 1
        for i in 1..<n {
            for j in 0..<i where check(j, i) {
                dp[i + 1] = max(dp[i + 1], dp[j + 1] + 1)
                ans = max(ans, dp[i + 1])
            }
        }
        func check(_ j: Int, _ i: Int) -> Bool {
            var diff = 0
            var k = 0, l = 0
            let n = words[j].count, m = words[i].count
            if m - n != 1 { return false }
            while k < n && l < m {
                if words[j][k] == words[i][l] {
                    k += 1
                    l += 1
                } else {
                    l += 1
                    diff += 1
                }
                if diff > 1 {
                    break
                }
            }
            diff += (m - l)
            return diff == 1
        }
        return ans
    }

    func maxTotalFruits(_ fruits: [[Int]], _ startPos: Int, _ k: Int) -> Int {
        let n = fruits.count
        var left = lowerBound(startPos - k), right = left
        var ans = 0, s = 0
        while right < n && fruits[right][0] <= startPos + k {
            s += fruits[right][1]
            while fruits[right][0] * 2 - fruits[left][0] - startPos > k && fruits[right][0] - fruits[left][0] * 2 + startPos > k {
                s -= fruits[left][1]
                left += 1
            }
            ans = max(ans, s)
            right += 1
        }
        func lowerBound(_ target: Int) -> Int {
            var l = 0, r = n
            while l < r {
                let mid = (l + r) >> 1
                if fruits[mid][0] >= target {
                    r = mid
                } else {
                    l = mid + 1
                }
            }
            return l
        }
        return ans
    }

    func hardestWorker(_ n: Int, _ logs: [[Int]]) -> Int {
        var arr = [Int](repeating: 0, count: n)
        var ans = 0
        for (i, log) in logs.enumerated() {
            arr[log[0]] = max(log[1] - (i == 0 ? 0 : logs[i - 1][1]), arr[log[0]])
        }
        for i in 0..<n where arr[i] > arr[ans] {
            ans = i
        }
        return ans
    }

    func minNumberOfFrogs(_ croakOfFrogs: String) -> Int {
        var idx: [Character: Int] = ["c": 0, "r": 1, "o": 2, "a": 3, "k": 4]
        var cnts = [Int](repeating: 0, count: 5)
        for c in croakOfFrogs {
            let i = idx[c]!
            if cnts[(i + 4) % 5] > 0 {
                cnts[(i + 4) % 5] -= 1
            } else if c != "c" {
                return -1
            }
            cnts[i] += 1
        }
        if cnts[0] > 0 || cnts[1] > 0 || cnts[2] > 0 || cnts[3] > 0 {
            return -1
        }
        return cnts[4]
    }

    func distinctDifferenceArray(_ nums: [Int]) -> [Int] {
        let n = nums.count
        var ans = [Int](repeating: 0, count: nums.count)
        for i in 0..<n {
            var set1 = Set<Int>()
            var set2 = Set<Int>()
            for j in 0...i {
                set1.insert(nums[j])
            }
            for k in i+1..<n {
                set2.insert(nums[k])
            }
            ans[i] = set1.count - set2.count
        }
        return ans
    }

    func colorTheArray(_ n: Int, _ queries: [[Int]]) -> [Int] {
        let m = queries.count
        var ans = [Int](repeating: 0, count: m)
        var arr = [Int](repeating: 0, count: n)
        var last = 0
        for (i, query) in queries.enumerated() {
            if query[0] > 0 && arr[query[0]] != 0 && arr[query[0]] == arr[query[0] - 1] {
                last -= 1
            }
            if query[0] < n - 1 && arr[query[0]] != 0 && arr[query[0]] == arr[query[0] + 1] {
                last -= 1
            }
            arr[query[0]] = query[1]
            if query[0] > 0 && arr[query[0]] == arr[query[0] - 1] {
                last += 1
            }
            if query[0] < n - 1 && arr[query[0]] == arr[query[0] + 1] {
                last += 1
            }
            ans[i] = last
        }
        return ans
    }

    func minIncrements(_ n: Int, _ cost: [Int]) -> Int {
        func dfs(i: Int) -> (Int, Int) {
            if i * 2 > n {
                return (cost[i - 1], 0)
            }
            let left = dfs(i: i * 2)
            let right = dfs(i: i * 2 + 1)
            let diff = (left.0 > right.0 ? left.0 - right.0 : right.0 - left.0) + left.1 + right.1
            return (max(left.0, right.0) + cost[i - 1], diff)
        }
        return dfs(i: 1).1
    }

    func countTime(_ time: String) -> Int {
        var chars = [Character](time.filter { $0 != ":" })
        func dfs(_ i: Int) -> Int {
            if i >= 4 {
                return check(chars) ? 1 : 0
            }
            if chars[i] == "?" {
                var ans = 0
                for i in 0...9 {
                    chars[i] = Character("\(i)")
                    ans += dfs(i + 1)
                    chars[i] = "?"
                }
                return ans
            }
            return dfs(i + 1)
        }
        func check(_ chars: [Character]) -> Bool {
            let hour = Int("\(chars[0])\(chars[1])")!
            let minute = Int("\(chars[2])\(chars[3])")!
            return hour >= 0 && hour <= 23 && minute >= 0 && minute <= 59
        }
        return dfs(0)
    }

    func smallestRepunitDivByK(_ k: Int) -> Int {
        if k % 2 == 0 || k % 5 == 0 {
            return -1
        }
        var x = 1 % k
        for i in 1...k {
            if x == 0 {
                return i
            }
            x = (x * 10 + 1) % k
        }
        return -1
    }

    func queryString(_ s: String, _ n: Int) -> Bool {
        var seen = Set<Int>()
        let chars = [Character](s), m = chars.count
        for i in 0..<m {
            var x = chars[i].wholeNumberValue!
            if x == 0 { continue }
            var j = i + 1
            while x <= n {
                seen.insert(x)
                if j == m { break }
                x = (x << 1) | chars[j].wholeNumberValue!
                j += 1
            }
        }
        return seen.count == n
    }

    func maxValueAfterReverse(_ nums: [Int]) -> Int {
        var base = 0, d = 0, n = nums.count
        var mx = Int.min, mn = Int.max
        for i in 1..<n {
            let a = nums[i - 1], b = nums[i]
            let dab = abs(a - b)
            base += dab
            mx = max(mx, min(a, b))
            mn = min(mn, max(a, b))
            d = max(d, abs(nums[0] - b) - dab, abs(nums[n - 1] - a) - dab)
        }
        return base + max(d, 2 * (mx - mn))
    }

    func maxEqualRowsAfterFlips(_ matrix: [[Int]]) -> Int {
        let m = matrix[0].count
        var ans = 0, map = [String: Int]()
        for row in matrix {
            var chars = row
            for i in 0..<m {
                chars[i] = chars[i] ^ row[0]
            }
            let str = chars.map({ String($0) }).joined()
            map[str, default: 0] += 1
            ans = max(ans, map[str, default: 0])
        }
        return ans
    }

    func minDifficulty(_ jobDifficulty: [Int], _ d: Int) -> Int {
        let n = jobDifficulty.count
        if n < d { return -1 }
        var memo = [[Int]](repeating: [Int](repeating: -1, count: d + 1), count: n)
        func dfs(i: Int, cur: Int) -> Int {
            if i >= n && cur > d {
                return 0
            } else if i >= n || cur > d {
                return 100000000
            }
            if memo[i][cur] != -1 {
                return memo[i][cur]
            }
            var res = jobDifficulty[i] + dfs(i: i + 1, cur: cur + 1)
            var first = jobDifficulty[i]
            var j = i + 1
            while j < n {
                first = max(first, jobDifficulty[j])
                res = min(res, dfs(i: j + 1, cur: cur + 1) + first)
                j += 1
            }
            memo[i][cur] = res
            return memo[i][cur]
        }
        return dfs(i: 0, cur: 1)
    }

    func haveConflict(_ event1: [String], _ event2: [String]) -> Bool {
        return !(event1[1] < event2[0] ||  event2[1] < event1[0])
    }

    func addNegabinary(_ arr1: [Int], _ arr2: [Int]) -> [Int] {
        let n = arr1.count, m = arr2.count
        let arr1 = [Int](arr1.reversed()), arr2 = [Int](arr2.reversed())
        var arr = [Int]()
        var firstCarry = 0, secondCarry = 0
        var i = 0
        while (firstCarry != 0 || secondCarry != 0) || (i < n || i < m) {
            var curr = secondCarry
            if i < n {
                curr += arr1[i]
            }
            if i < m {
                curr += arr2[i]
            }
            arr.append(curr & 1)
            secondCarry = (curr >> 1) + firstCarry
            firstCarry = curr >> 1
            i += 1
            while firstCarry >= 1 && secondCarry >= 2 {
                firstCarry -= 1
                secondCarry -= 2
            }
        }
        var ans = [Int](), flag = arr.count > 1
        for num in arr.reversed() {
            if num == 0 && flag {
                continue
            } else {
                flag = false
                ans.append(num)
            }
        }
        if ans.count == 0 { ans = [0] }
        return ans
    }

    func numTilePossibilities(_ tiles: String) -> Int {
        let n = tiles.count, chars = [Character](tiles)
        var set = Set<String>()
        var visvited = [Bool](repeating: false, count: n)
        func dfs(i: Int, curr: String) {
            set.insert(curr)
            for k in 0..<n {
                if visvited[k] { continue }
                visvited[k] = true
                dfs(i: k + 1, curr: "\(curr)\(chars[k])")
                visvited[k] = false
            }
        }
        dfs(i: 0, curr: "")
        return set.count - 1
    }

    func maxSumBST(_ root: TreeNode?) -> Int {
        let inf = 0x3F3F3F3F
        var res = 0
        @discardableResult
        func dfs(curr: TreeNode?) -> (Bool, Int, Int, Int) {
            guard let curr = curr else { return (true, inf, -inf, 0) }
            let left = dfs(curr: curr.left)
            let right = dfs(curr: curr.right)
            if left.0 && right.0 && curr.val > left.2 && curr.val < right.1 {
                let sum = curr.val + left.3 + right.3
                res = max(res, sum)
                return (true, min(left.1, curr.val), max(right.2, curr.val), sum)
            }
            return (false, 0, 0, 0)
        }
        dfs(curr: root)
        return res
    }

    override var excuteable: Bool { return true }

    override func executeTestCode() {
//        super.executeTestCode()
    }
}

class FrequencyTracker {
    var map1 = [Int: Int]()
    var map2 = [Int: Set<Int>]()

    init() { }

    func add(_ number: Int) {
        let cnt = map1[number, default: 0]
        map1[number, default: 0] += 1
        map2[cnt, default: Set<Int>()].remove(number)
        map2[cnt + 1, default: Set<Int>()].insert(number)
    }

    func deleteOne(_ number: Int) {
        if let cnt = map1[number], cnt > 0 {
            map1[number]! -= 1
            map2[cnt, default: Set<Int>()].remove(number)
            map2[cnt - 1, default: Set<Int>()].insert(number)
        }
    }

    func hasFrequency(_ frequency: Int) -> Bool {
        return (map2[frequency]?.count ?? 0) > 0
    }
}
