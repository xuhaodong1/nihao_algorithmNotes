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

    /// 题目链接：[669. 修剪二叉搜索树](https://leetcode.cn/problems/trim-a-binary-search-tree/)
    func trimBST(_ root: TreeNode?, _ low: Int, _ high: Int) -> TreeNode? {
        func dfs(_ node: TreeNode?) -> TreeNode? {
            guard let node = node else { return nil }
            if node.val < low {
                return dfs(node.right)
            } else if node.val > high {
                return dfs(node.left)
            }
            node.left = dfs(node.left)
            node.right = dfs(node.right)
            return node
        }
        return dfs(root)
    }

//    override var excuteable: Bool {
//        return true
//    }

    override func executeTestCode() {
        super.executeTestCode()
    }
}
