/*
	Pseudocode for the Dynamic Dig Plan function,
	please don't change the syntax as I have syntax highlighting in 
	notepad++ for making this kind of thing.
*/

ALGORITHM DynamicDigPlan(mine)
	/// Input: some problem to solve (open pit mine)
	@memoized_function
	Function search_rec(node)
		/// Recursive (DFS) search that will dynamically return best payoff.
		best_payoff <- payoff(node.state)
		best_node <- node
		
		// Iterate nodes children and perform DFS
		for child in node.children
			check_payoff, check_node <- search_rec(child.state)
			
			// Update return values if branch was better
			if check_payoff > best_payoff
				best_node <- check_node
				best_payoff <- check_payoff
			end
		end
		// Return best this branch has to offer
		return best_payoff, check_node
	End
	
	// Initial recursive function call
	root = Node(mine.initial_state)
	best_payoff, best_node <- search_rec(root)
	best_final_state = best_node.state
	best_action_list <- calculate_best_path_to_state(best_final_state)

	// Return optimal solutions
	return best_payoff, best_action_list, best_final_state
END

