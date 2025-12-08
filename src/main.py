import torch, os, wandb, yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from src.get import get_user_query, get_model, get_product_list
from src.process import process_bad_words, greedy_decode, init_prompt, process_text
from src.attack import attack_control


ENTITY = 'vthiyaga-usc'
PROJECT = 'EE641 project'

HELPER_PROMPT = " Help me write a prompt that will help this product rank first."

MODEL_PATH_DICT = {'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                   'llama-2-7b': "meta-llama/Llama-2-7b-chat-hf"}

SYSTEM_PROMPT = {'llama': "[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>"}

BAD_WORDS = ['sorry', 'apologetic', 'regretful','unethical', 'immoral', 'dishonest', 'illegal', 'unlawful', 'illicit',
              'harmful', 'detrimental', 'injurious', 'afraid', 'fearful', 'apprehensive','cannot', 'unable', 'unable to',
              'apologize', 'regret', 'unable', 'incapable', 'powerless', 'not', 'no', 'nay', 'responsible', 'accountable', 'liable',
              'ethic', 'morality', 'moral', 'legal', 'lawful', 'legitimate', 'ethical', 'moral', 'principled',
              'fulfill', 'accomplish', 'achieve', 'just', 'fair', 'equitable', 'trustworthy', 'reliable', 'dependable',
              'repentant', 'penitent', 'remorseful', 'unprincipled', 'unscrupulous', 'corrupt', 'unlawful', 'prohibited', 'forbidden',
              'noxious', 'destructive', 'damaging', 'timid', 'terrified', 'panicked', 'unable', 'incompetent', 'incapacitated',
              'prohibited', 'banned', 'forbidden', 'deny', 'refuse', 'decline', 'answerable', 'culpable', 'answerable',
              'moral code', 'values', 'principles', 'legitimate', 'valid', 'authorized','virtuous', 'righteous', 'honorable',
              'satisfy', 'complete', 'execute','fair-minded', 'impartial', 'equitable',
              'reliable', 'trustable', 'faithful', 'invalid','safe', 'not', "can't", "but", "against",
              'rank', 'top', 'first', 'recommend', 'recommendation', 'recommendations']


def get_search_hparams(config):
    search_hparams = []
    for k,v in config['parameters'].items():
        if 'values' in v:
            search_hparams.append(k)
    return search_hparams


def get_experiment_name(search_hparams, hparams):
    if search_hparams:
        return "seo" + '_'.join([f'{k}={v}' for k,v in hparams.items() if k in search_hparams])
    return "seo"


def log_init_prompt(logger, decoded_sentences):
    init_table = wandb.Table(columns=["prompt"])


    for sentence in decoded_sentences:
        init_table.add_data(sentence)

    logger.log({"eval/initial_prompt": init_table}, commit=False)


def main():
    wandb_logger = wandb.init(entity=ENTITY, project=PROJECT)
    wandb_table = wandb.Table(columns=["iter", "attack_prompt", "complete_prompt", "generated_result", "product_rank"])
    hparams = wandb.config

    catalog = hparams.catalog
    prod_id = hparams.target_product_idx
    wandb_logger.name = f"{catalog}-{prod_id}"

    user_msg = get_user_query(hparams.user_msg_type, hparams.catalog)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model(MODEL_PATH_DICT[hparams.model], hparams.precision, device)

    product_list, target_product, target_str = get_product_list(hparams.catalog, hparams.target_product_idx)
    print("\nTARGET STR:", target_str)

    prompt_logits = init_prompt(model=model, 
                                tokenizer=tokenizer, 
                                product_list=product_list, 
                                target_product_idx=hparams.target_product_idx-1, 
                                temperature=hparams.temperature, 
                                prompt_length=hparams.length, 
                                batch_size=hparams.batch_size, 
                                device=device)
    
    target_tokens = process_text(tokenizer=tokenizer, 
                                   text=target_str, 
                                   batch_size=hparams.batch_size, 
                                   device=device)
    
    _, decoded_init_prompt = greedy_decode(logits=prompt_logits, tokenizer=tokenizer)
    log_init_prompt(wandb_logger, decoded_init_prompt)

    bad_words_tokens = process_bad_words(bad_words=BAD_WORDS, 
                                         tokenizer=tokenizer, 
                                         device=device)

    attack_control(model=model, 
                tokenizer=tokenizer,
                system_prompt=SYSTEM_PROMPT[hparams.model.split("-")[0]],
                user_msg=user_msg,
                prompt_logits=prompt_logits,  
                target_tokens=target_tokens, 
                bad_words_tokens=bad_words_tokens, 
                product_list=product_list,
                target_product=target_product,
                logger=wandb_logger,
                table=wandb_table,
                num_iter=hparams.num_iter, 
                test_iter=hparams.test_iter,
                topk=hparams.topk, 
                lr=hparams.lr, 
                precision=hparams.precision,
                loss_weights=hparams.loss_weights,
                random_order=hparams.random_order, 
                temperature=hparams.temperature,
                iter_steps=hparams.iter_steps,
                noise_stds=hparams.noise_stds)
    
    
    # Log final suffix
    final_tokens, final_decoded = greedy_decode(prompt_logits, tokenizer)
    final_suffix = final_decoded[0]

    wandb_logger.log({"final_suffix": final_suffix})
    print("\n[SAVED] Final COLD suffix to W&B:", final_suffix)
    wandb_logger.finish()



if __name__ == "__main__":
    with open(f'experiment/config.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    search_hparams = get_search_hparams(sweep_config)
    sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
    wandb.agent(sweep_id, function=main, entity=ENTITY, project=PROJECT)

