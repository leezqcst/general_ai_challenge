import sonnet as snt

MASTER_CONTROLLER=-1
NUM_RNN_STATES=2
NUM_STATES=NUM_RNN_STATES
class RNNRecurrentBoltzmannMachine(snt.RNNCore):
	def __init__(self,num_cluster_layers,cluster_layer_size,hidden_dim,output_dim,unique_embedding=True,
		linearity_before=True,linearity_after=True,embed_controller=True,module_type='RNN',name='rnn_rbm'):
		assert module_type=='RNN'
		super(RecurrentBoltzmannMachine,self).__init__(name=name)
		assert num_cluster_layers>0
		self._num_cluster_layers=num_cluster_layers
		assert cluster_layer_size>0
		self._cluster_layer_size=cluster_layer_size
		assert hidden_dim>0
		self._hidden_dim=hidden_dim
		self._unique_embedding=unique_embedding
		self._linearity_before=linearity_before
		self._linearity_after=linearity_after
		self._module_type=module_type
		self._embed_controller=embed_controller
		self._output_dim=output_dim
		self._create_structure()

	def _create_structure(self):
		curr_inputs=inputs
		self._controllers=[]
		self._input_linearities=[]
		self._output_linearities=[]
		self._submodules=[]
		for layer in range(self._num_cluster_layers):
			master_controller=snt.RNN(self._hidden_dim,name='master_controller_'+str(layer))
			master_controller_weight_encoder=snt.Linear(self._cluster_layer_size*self._cluster_layer_size,name='master_encoder_'+str(layer))
			if self._embed_controller:
				master_controller_embedding=snt.Linear(self._hidden_dim,name='master_controller_'+str(layer)+'_embedding')
				to_append=(master_controller,master_controller_embedding,master_controller_weight_encoder)
			else:
				to_append=(master_controller,master_controller_weight_encoder)

			self._controllers.append(to_append)
			current_input_linearities=[]
			current_output_linearities=[]
			current_submodules=[]
			if not self._unique_embedding and self._linearity_before:
				#TODO: mikght need a call to reuse_variables
				linear_in=snt.Linear(output_size=self._hidden_dim,name='embedding_'+str(layer))
				current_input_linearities.append(linear_in)
			if not self._unique_embedding and self._linearity_after:
				#TODO: mikght need a call to reuse_variables
				linear_out=snt.Linear(output_size=self._hidden_dim,name='subencoder_'+str(layer))
				current_output_linearities.append(linear_out)
			for submodule in range(self._cluster_layer_size):
				if self._unique_embedding and self._linearity_before:
					linear_in=snt.Linear(output_size=self._hidden_dim,name='subembedding_'+str(layer)+'_'+str(submodule))
					current_input_linearities.append(linear_in)
				submodule=snt.RNN(self._hidden_dim,name='submodule_'+str(layer)+'_'+str(submodule))
				if self._unique_embedding and self._linearity_after:
					linear_out=snt.Linear(output_size=self._hidden_dim,name='subencoder_'+str(layer)+'_'+str(submodule))
					current_output_linearities.append(linear_out)
				current_submodules.append(submodule)
			self._output_linearities.append(current_output_linearities)
			self._input_linearities.append(current_input_linearities)
			self._submodules.append(current_submodules)
		self._master_encoder=snt.Linear(output_size=self._output_dim)

	@property
	def state_size(self):
		state_size=[]
		for i in range(self._num_cluster_layers):
			for j in range(self._cluster_layer_size+1):
				for x in range(NUM_STATES):
					state_size.append(tf.TensorShape([self._hidden_dim]))
		return tuple(state_size)

	def get_state(self,states,layer_num,submodule):
		module_number=submodule+1
		assert layer_num<self._num_cluster_layers
		assert submodule<self._cluster_layer_size
		index_number=(self._cluster_layer_size+1)*layer_num+module_number
		index_number*=NUM_STATES
		the_state=[]
		for i in range(NUM_STATES):
			the_state.append(states[index_number+i])
		return tuple(the_state)


	def _build(self,inputs,state):
		curr_inputs=inputs
		new_states=[]
		module_output=None
		for layer in range(self._num_cluster_layers):
			
			if self._embed_controller:
				master_controller,master_controller_embedding,encoder=self._controllers[layer]
				master_in=master_controller_embedding(inputs)
				transformation_weights,new_state=master_controller(master_in,self.get_state(state,layer,MASTER_CONTROLLER))
				transformation_weights=encoder(transformation_weights)
				new_states.append(new_state)
			else:
				master_controller,encoder=self._controllers[layer]
				transformation_weights,new_state=master_controller(inputs,self.get_state(state,layer,MASTER_CONTROLLER))
				transformation_weights=encoder(transformation_weights)
				new_states.append(new_state)

			if not self._unique_embedding and self._linearity_before:
				#TODO: mikght need a call to reuse_variables
				linear_in=self._input_linearities[layer][0]
			if not self._unique_embedding and self._linearity_after:
				#TODO: mikght need a call to reuse_variables
				linear_out=self._output_linearities[layer][0]
			if layer==0:
				curr_inputs=[curr_inputs*self._cluster_layer_size]
			module_outputs=[]
			curr_states=[]
			for submodule in range(self._cluster_layer_size):
				if self._unique_embedding and self._linearity_before:
					linear_in=self._input_linearities[layer][submodule]
				if self._linearity_before:
					submodule_input=linear_in(curr_inputs[submodule])
				else:
					curr_inputs[submodule]
				lstm_output,new_state=self._submodules[layer][submodule](submodule_input,self.get_state(state,layer,submodule))
				curr_states.append(new_state)
				if self._unique_embedding and self._linearity_after:
					linear_out=self._output_linearities[layer][submodule]
				if self._linearity_after:
					submodule_output=linear_out(lstm_output)
				else:
					submodule_output=lstm_output
				module_outputs.append(tf.expand_dims(submodule_output,-1))
			curr_inputs=module_outputs
			module_outputs=tf.concat(module_outputs,axis=-1)
			module_output=module_outputs.reshape(self._hidden_dim*self._cluster_layer_size)
			module_outputs=tf.expand_dims(module_outputs,axis=-1)
			#######
			# perform weighted master_controller op here and feed it to next layer as curr_inputs (curr_inputs should be shape [self._cluster_layer_size,hidden_dim])
			# TODO
			# TODO
			# TODO
			#######
			transformation_weights=tf.reshape([-1,1,self._cluster_layer_size,self._cluster_layer_size])
			normalized_weights=tf.nn.softmax(transformation_weights)
			state_updates=tf.multiply(module_outputs,normalized_weights)
			state_updates=tf.reduce_sum(state_updates,axis=-2)
			state_updates=tf.squeeze()
			for i in range(self._cluster_layer_size):
				curr_states[i]+=state_updates[:,:,i]
			new_states+=curr_states
		return self._master_encoder(module_output),tuple(new_states)


					
