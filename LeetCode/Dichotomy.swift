//
//  Dichotomy.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/12.
//

import Foundation

/// 二分相关练习题
class Dichotomy: BaseCode {

    /// 题目链接：[1608. 特殊数组的特征值](https://leetcode.cn/problems/special-array-with-x-elements-greater-than-or-equal-x/)
    func specialArray(_ nums: [Int]) -> Int {
        let n = nums.count, nums = nums.sorted()
        var l = 0, r = Int.max
        while l < r {
            let mid = l + (r - l) >> 1
            if mid >= getCnt(x: mid) {
                r = mid
            } else {
                l = mid + 1
            }
        }
        func getCnt(x: Int) -> Int {
            var l = 0, r = n - 1
            while l < r {
                // mid = (l + r) >> 1 每次找到左边, 因此需要构造 l = mid + 1
                // 如 l = 0, r = 1, 则 mid = 0, 执行 if 结果为 l = mid, 则会造成死循环
                let mid = (l + r) >> 1
                if nums[mid] >= x {
                    r = mid
                } else {
                    l = mid + 1
                }
            }
            return nums[l] >= x ? n - l : 0 // 边界判断
        }
        return getCnt(x: l) == l ? l : -1
    }

//    override var excuteable: Bool {
//        return true
//    }

    override func executeTestCode() {
        super.executeTestCode()
    }

}
