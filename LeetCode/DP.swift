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

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(jobScheduling([1,1,1], [2,3,4], [5,6,4]))
    }
}
