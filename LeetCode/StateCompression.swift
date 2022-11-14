//
//  State Compression.swift
//  LeetCode
//
//  Created by haodong xu on 2022/11/14.
//

import Foundation

/// 状态压缩相关练习题
class StateCompression: BaseCode {

    /// 题目链接：[805. 数组的均值分割](https://leetcode.cn/problems/split-array-with-same-average/description/)
    /// 折半搜索 + 状态压缩
    func splitArraySameAverage(_ nums: [Int]) -> Bool {
        let n = nums.count, m = n / 2, sum = nums.reduce(0, +)
        var map = [Int: Set<Int>]()
        for s in 0..<(1 << m) {
            var tot = 0, cnt = 0
            for i in 0..<m where ((s >> i) & 1) == 1 {
                tot += nums[i]
                cnt += 1
            }
            var set: Set<Int> = map[tot, default: Set<Int>()]
            set.insert(cnt)
            map[tot] = set
        }
        for s in 0..<(1 << (n - m)) {
            var tot = 0, cnt = 0
            for i in 0..<(n - m) where ((s >> i) & 1) == 1 {
                tot += nums[i + m]
                cnt += 1
            }
            for k in max(1, cnt)..<n {
                if (k * sum) % n != 0 { continue }
                let t = k * sum / n
                if !map.keys.contains(t - tot) { continue }
                if !(map[t - tot]?.contains(k - cnt) ?? false) { continue }
                return true
            }
        }
        return false
    }

}
