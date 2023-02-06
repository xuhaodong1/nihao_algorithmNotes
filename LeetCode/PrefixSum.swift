//
//  PrefixSum.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/24.
//

import Foundation

/// 前缀和相关练习题
class PrefixSum: BaseCode {

    /// 题目链接：[1652. 拆炸弹](https://leetcode.cn/problems/defuse-the-bomb/)
    func decrypt(_ code: [Int], _ k: Int) -> [Int] {
        let n = code.count
        var ans = [Int](repeating: 0, count: n)
        var preSum = [Int](repeating: 0, count: 2 * n + 1)
        for i in 1...2*n {
            preSum[i] += preSum[i - 1] + code[(i - 1) % n]
        }
        for i in 1...n {
            if k > 0 { ans[i - 1] = preSum[i + k] - preSum[i] }
            else { ans[i - 1] = preSum[i + n - 1] - preSum[i + n + k - 1] }
        }
        return ans
    }

    /// 题目链接：[764. 最大加号标志](https://leetcode.cn/problems/largest-plus-sign/description/)
    func orderOfLargestPlusSign(_ n: Int, _ mines: [[Int]]) -> Int {
        let set = Set<[Int]>(mines)
        var ans = 0
        var arr = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: 4), count: n + 2), count: n + 2)
        for i in 1...n {
            for j in 1...n {
                if !set.contains([i - 1, j - 1]) {
                    arr[i][j][0] = arr[i - 1][j][0] + 1
                    arr[i][j][1] = arr[i][j - 1][1] + 1
                }
                let i2 = n + 1 - i, j2 = n + 1 - j
                if !set.contains([i2 - 1, j2 - 1]) {
                    arr[i2][j2][2] = arr[i2 + 1][j2][2] + 1
                    arr[i2][j2][3] = arr[i2][j2 + 1][3] + 1
                }
            }
        }
        (1...n).forEach { i in (1...n).forEach { j in ans = max(ans, arr[i][j].min()!) } }
        return ans
    }

    /// 题目链接：[1703. 得到连续 K 个 1 的最少相邻交换次数](https://leetcode.cn/problems/minimum-adjacent-swaps-for-k-consecutive-ones/description/)
    func minMoves(_ nums: [Int], _ k: Int) -> Int {
        var p = [Int]()
        for i in 0..<nums.count where nums[i] != 0 {
            p.append(i - p.count)
        }
        let m = p.count
        var s = [Int](repeating: 0, count: m + 1)
        for i in 0..<m {
            s[i + 1] = s[i] + p[i]
        }
        var ans = Int.max
        for i in 0...m-k {
            ans = min(ans, s[i] + s[i + k] - s[i + k / 2] * 2 - p[i + k / 2] * (k % 2))
        }
        return ans
    }

    /// 题目链接：[2559. 统计范围内的元音字符串数](https://leetcode.cn/problems/count-vowel-strings-in-ranges/description/)
    func vowelStrings(_ words: [String], _ queries: [[Int]]) -> [Int] {
        let n = words.count, m = queries.count
        let yuanYin: [Character] = ["a", "e", "i", "o", "u"]
        var ans = [Int](repeating: 0, count: m)
        var preSum = [Int](repeating: 0, count: n + 1)
        for i in 1...n {
            let curr = yuanYin.contains(words[i - 1].first!) && yuanYin.contains(words[i - 1].last!) ? 1 : 0
            preSum[i] = preSum[i - 1] + curr
        }
        for (i, query) in queries.enumerated() {
            ans[i] = preSum[query[1] + 1] - preSum[query[0]]
        }
        return ans
    }

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(decrypt([2,4,9,3],
                      -2))
    }
}
