
domain="san"
#####################################################
## DCST with Pretrained Layers
####################################################


# word_path="../self_training_old/data/multilingual_word_embeddings/cc.sanskrit.300.new.vec"
word_path="./data/morph.word.200.vec"
# word_path="./data/cc.sanskrit.300.FT.case.nos.vec"
char_path="./data/morph.char.30.vec"
pos_path='../self_training_old/data/multilingual_word_embeddings/pos_embedding_FT.100'  

declare -i num_epochs=100
declare -i word_dim=200
declare -i set_num_training_samples=500
start_time=`date +%s`
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
model_path="ud_parser_san_"$current_time 
touch saved_models/base_log.txt
## model_path="ud_parser_san_"

################################################################
# Running the base Biaffine Parser
echo "#################################################################"
echo "Currently base model in progress..."
echo "#################################################################"
python examples/GraphParser.py --dataset ud --domain $domain --rnn_mode LSTM \
--num_epochs $num_epochs --batch_size 16 --hidden_size 512 --arc_space 512  \
--arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --pos_embedding random \
--word_dim $word_dim --char_dim 30 --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 \
--epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst \
--set_num_training_samples $set_num_training_samples \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext --char_embedding hellwig --char_path $char_path --word_path $word_path \
--model_path saved_models/$model_path 2>&1 | tee saved_models/base_log.txt
mv saved_models/base_log.txt saved_models/$model_path/base_log.txt

# ###################################################################
## Running the Sequence Tagger
for task in 'number_of_children' 'relative_pos_based' 'distance_from_the_root'; do
	touch saved_models/$model_path/log.txt
	echo "#################################################################"
	echo "Currently $task in progress..."
	echo "#################################################################"  
    python examples/SequenceTagger.py --dataset ud --domain $domain --task $task \
    --rnn_mode LSTM --num_epochs $num_epochs --batch_size 16 --hidden_size 512 \
	--tag_space 128 --num_layers 3 --num_filters 100 --use_char  --char_dim 200 \
	--pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 \
	--schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 \
	--p_in 0.33 --p_out 0.33 --punct_set '.' '``'  ':' ','  \
	--word_dim $word_dim --word_embedding fasttext --word_path $word_path --pos_embedding random \
	--parser_path saved_models/$model_path/ \
	--use_unlabeled_data --char_embedding random --char_path $char_path \
	--model_path saved_models/$model_path/$task/ 2>&1 | tee saved_models/$model_path/log.txt
	mv saved_models/$model_path/log.txt saved_models/$model_path/$task/log.txt
done

#########################################################################
# Final step - Running the Combined DCST Parser
echo "#################################################################"
echo "Currently final model in progress..."
echo "#################################################################"
touch saved_models/$model_path/log.txt
# As a final step we can now run the DCST (ensemble) parser:
python examples/GraphParser.py --dataset ud --domain $domain --rnn_mode LSTM \
--num_epochs $num_epochs --batch_size 16 --hidden_size 512 \
--arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char  \
--word_dim $word_dim --char_dim 30 --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 \
--p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --pos_embedding random \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext --char_embedding hellwig --char_path $char_path --word_path $word_path  \
--gating --num_gates 4 \
--load_sequence_taggers_paths saved_models/$model_path/number_of_children/domain_$domain.pt \
saved_models/$model_path/relative_pos_based/domain_$domain.pt \
saved_models/$model_path/distance_from_the_root/domain_$domain.pt \
--model_path saved_models/$model_path/final_ensembled/ 2>&1 | tee saved_models/$model_path/log.txt
mv saved_models/$model_path/log.txt saved_models/$model_path/final_ensembled/log.txt

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

