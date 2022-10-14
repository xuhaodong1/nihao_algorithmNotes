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

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(distinctSubseqII("abc"))
    }
}
