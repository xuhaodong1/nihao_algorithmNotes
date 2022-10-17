//
//  SlidingWindow.swift
//  LeetCode
//
//  Created by haodong xu on 2022/10/17.
//

import Foundation

/// 滑动窗口相关练习题
class SlidingWindow: BaseCode {

    /// 题目链接：[904. 水果成篮](https://leetcode.cn/problems/fruit-into-baskets/)
    func totalFruit(_ fruits: [Int]) -> Int {
        let n = fruits.count
        var map = [Int: Int](), ans = 0
        var l = 0, r = 0
        while r < n {
            map[fruits[r], default: 0] += 1
            while l < r && map.count > 2 {
                map[fruits[l]]! -= 1
                if map[fruits[l]] == 0 { map.removeValue(forKey: fruits[l]) }
                l += 1
            }
            ans = max(ans, r - l + 1)
            r += 1
        }
        return ans
    }

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(totalFruit([1,2,1]))
    }
}
