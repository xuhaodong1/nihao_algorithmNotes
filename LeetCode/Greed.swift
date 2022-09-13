//
//  Greed.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/3.
//

import Foundation

/// 贪心相关练习题
class Greed: BaseCode {

    /// 题目链接：[646. 最长数对链](https://leetcode.cn/problems/maximum-length-of-pair-chain/)
    func findLongestChain(_ pairs: [[Int]]) -> Int {
        let pairs = pairs.sorted { pair1, pair2 in
            return pair1[1] < pair2[1]
        } /// 贪心思想, 最优的选择是挑选的第二个数字最小的, 这样后续挑选的数对留下更多的空间
        var ans = 0, curr = Int.min
        for pair in pairs where curr < pair[0] {
            curr = pair[1]
            ans += 1
        }
        return ans
    }

    /// 题目链接：[670. 最大交换](https://leetcode.cn/problems/maximum-swap/)
    /// 思路：贪心, 从左往右找到每一位的左边最大值, 之后从右往左查看其左边是否有最大值
    func maximumSwap(_ num: Int) -> Int {
        var digits = "\(num)".reversed().map({ return Int("\($0)")! })
        let n = digits.count
        var maxIdxs = [Int](repeating: -1, count: n), maxIdx = 0
        for i in 0..<n {
            if digits[i] < digits[maxIdx] {
                maxIdxs[i] = maxIdx
            } else if digits[i] > digits[maxIdx] { // == 时不更新, 应保持最大值下标尽量靠左
                maxIdx = i
            }
        }
        for i in (0..<n).reversed() where maxIdxs[i] != -1 {
            digits.swapAt(i, maxIdxs[i])
            break
        }
        return Int(String(digits.map({ return Character("\($0)") }).reversed()))!
    }

}
