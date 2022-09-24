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

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(decrypt([2,4,9,3],
                      -2))
    }
}
