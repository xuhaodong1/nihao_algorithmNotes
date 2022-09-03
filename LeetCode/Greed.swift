//
//  Greed.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/3.
//

import Cocoa

class Greed {

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

}
