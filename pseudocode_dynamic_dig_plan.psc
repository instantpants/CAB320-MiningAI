/*
	Pseudocode for the Dynamic Dig Plan function,
	please don't change the syntax as I have syntax highlighting in 
	notepad++ for making this kind of thing.
*/

ALGORITHM DynamicDigPlan(mine)
	/// Input: some problem to solve (open pit mine)
	
	@memoized_function
	Function search_rec(node)
		/// Recursive (DFS) search function that will dynamically retuen best payoff.
		
		// Recursive variables
		best_payoff <- payoff(node.state)
		best_state <- node.state
		
		// Search state branches
		for child in node.children
			check_payoff, check_state <- search_rec(child.state)
			
			// If branch is better update recursive variables
			if check_payoff > best_payoff
				best_state <- check_state
				best_payoff <- check_payoff
			end
		end
		// Return best the branch has to offer
		return best_payoff, best_state
	End
	
	// Initial recursive function call
	root = Node(mine.initial)
	best_payoff, best_final_state <- search_rec(root)
	
	// Function to calculate best path to state
	best_action_list <- calculate_path_to_state(best_final_state)
	
	return best_payoff, best_action_list, best_final_state
END

