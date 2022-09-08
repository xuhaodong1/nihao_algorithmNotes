//
//  Construction.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/8.
//

import Foundation

/// 构造题相关
class Construction: BaseCode {

    /// 题目链接：[667. 优美的排列 II](https://leetcode.cn/problems/beautiful-arrangement-ii/)
    func constructArray(_ n: Int, _ k: Int) -> [Int] {
        var ans = [Int]()
        var forward = true
        var start = 1, end = k + 1
        while start <= end {
            if forward {
                ans.append(start)
                start += 1
            } else {
                ans.append(end)
                end -= 1
            }
            forward = !forward
        }
        if k + 2 <= n {
            for i in k+2...n {
                ans.append(i)
            }
        }
        return ans
    }

//    override var excuteable: Bool {
//        return true
//    }

    override func executeTestCode() {
        super.executeTestCode()
    }
}
