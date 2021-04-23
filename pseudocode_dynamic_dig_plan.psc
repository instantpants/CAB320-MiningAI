/*
	Pseudocode for the Dynamic Dig Plan function,
	please don't change the syntax as I have syntax highlighting in 
	notepad++ for making this kind of thing.

	Thought I'd upload it incase you want something to work off when
	doing the BB pseudocode (this is the industry standard for writing
	pseudocode)
*/

ALGORITHM DynamicDigPlan(mine)
	@memoized_function
	Function search_rec(state)
		// Recursive variables
		best_payoff <- payoff(state)
		best_state <- state
		
		// Search state branches
		for action in possible_actions
			child_state <- state_after_action
			check_payoff, check_state <- search_rec(child_state)
			
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
	best_payoff, best_final_state <- search_rec(initial_state)
	
	// Function to calculate best path to state
	best_action_list <- calculate_path_to_state(best_final_state)
	
	return best_payoff, best_action_list, best_final_state
END

