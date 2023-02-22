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

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(stoneGameII([1]))
    }
}
