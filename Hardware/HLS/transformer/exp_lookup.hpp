#ifndef __EXP_LOOKUP__HPP__

#define __EXP_LOOKUP__HPP__

#include "exp_lookup_table.hpp"


template<typename TI, typename TO>
void ExpLookup(TI elem_in, TO &elem_out)
{

	TO elem_out_buffer;
	if(elem_in <= TI(lower_limit))
		{elem_out_buffer = exp_lookup_table[0];}
	else if (elem_in >= TI(upper_limit))
		{elem_out_buffer = exp_lookup_table[NUM_ELEMS-1];}
	else
	{
		TO zero_point = elem_in - TI (lower_limit);
		uint16_t idx = zero_point * recip_step;
		elem_out_buffer = exp_lookup_table[idx];
	}
	elem_out = elem_out_buffer;
}

#endif
