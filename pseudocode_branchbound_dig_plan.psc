/*
	Pseudocode for the Branch and Bound Dig Plan function,
	please don't change the syntax as I have syntax highlighting in 
	notepad++ for making this kind of thing.
*/

ALGORITHM BranchAndBoundDigPlan(mine)
	/// Input: problem to solve (an open ground mine)
	
	@memoized_function
	Function b(state)
		/// Returns best possible sum from some state when ignoring slope constraint
		best_branch_cumsum = max(mine.cumsum_mine where z >= state.z on z axis)
		return sum(best_branch_cumsum)
	End
	
	@memoized_function
	Function expand(node)
		/// Returns all children of some node, where b(s) > best_payoff
		return node.children where b(child.state) > best_payoff
	End
	
	// Initialize root node and frontier
	root = Node(mine.initial_state)
	frontier << root
	
	while frontier.has_value
		node = frontier.pop()
		payoff = node.payoff
		
		if payoff >= best_payoff
			best_payoff <- payoff
			best_node <- node
			
			// Prune all other branches and reinitialize frontier
			frontier <- expand(node)
		else
			// Add all children of node to frontier
			frontier + expand(node)
		end
	end
	
	// 
	best_final_state <- best_node.state
	best_action_list <- calculate_path_to_state(best_final_state)
	
	return best_payoff, best_action_list, best_final_state
END