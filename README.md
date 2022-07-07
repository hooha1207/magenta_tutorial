# magenta_tutorial
<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/hooha1207/magenta_tutorial.git"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Colab</a>
  </td>
</table>

<br/>
<br/>
<br/>

Colab 파일에 목차 기능을 이용해서 좀 더 편하게 볼 수 있도록 해놨습니다  
Colab으로 이동해서 면 감사하겠습니다


# cmd로 midi 파일 전처리 및 checkpoint 만드는 방법 정리
### 만약 custom checkpoint 및 tfrecord를 다 만들었다면, <br/> magenta_musicvae_4bar_code.py 를 수정해서 바로 실행하면 output 제작 가능합니다

### output 폴더에서 custom dataset으로 만든 midi 파일을 확인할 수 있습니다
before :

<br/>

open cmd  
create python 3.6.x version virtual environment  
pip install magenta  

<br/>

## [sample command]

music_vae_generate \ --config=cat-drums_2bar_small.hikl \ --checkpoint_file=C:/magenta_tutorial/checkpoints/cat-drums_2bar_small.hikl.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=cat-drums_2bar_small.lokl \ --checkpoint_file=C:/magenta_tutorial/checkpoints/cat-drums_2bar_small.lokl.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=cat-mel_2bar_big \ --checkpoint_file=C:/magenta_tutorial/checkpoints/cat-mel_2bar_big.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=groovae_2bar_add_closed_hh \ --checkpoint_file=C:/magenta_tutorial/checkpoints/groovae_2bar_add_closed_hh.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=groovae_2bar_hits_control \ --checkpoint_file=C:/magenta_tutorial/checkpoints/groovae_2bar_hits_control.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=groovae_2bar_humanize \ --checkpoint_file=C:/magenta_tutorial/checkpoints/groovae_2bar_humanize.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=groovae_2bar_tap_fixed_velocity \ --checkpoint_file=C:/magenta_tutorial/checkpoints/groovae_2bar_tap_fixed_velocity.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=groovae_4bar \ --checkpoint_file=C:/magenta_tutorial/checkpoints/groovae_4bar.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=hierdec-mel_16bar \ --checkpoint_file=C:/magenta_tutorial/checkpoints/hierdec-mel_16bar.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=hierdec-trio_16bar \ --checkpoint_file=C:/magenta_tutorial/checkpoints/hierdec-trio_16bar.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output

music_vae_generate \ --config=nade-drums_2bar_full \ --checkpoint_file=C:/magenta_tutorial/checkpoints/nade-drums_2bar_full.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output


run_dir  
checkpoint_file  
output_dir = 'tmp/music_vae/generated'  
config  
mode = sample  
input_midi_1  
input_midi_2  
num_outputs = 5  
max_batch_size = 8  
temperature = 0.5  
log = INFO  

<br/>
<br/>

## [ interpolation command]

music_vae_generate --config=cat-mel_2bar_big --checkpoint_file=C:/magenta_tutorial/checkpoints/groovae_4bar.tar --mode=interpolate --num_outputs=5 --input_midi_1=C:/Users/USER/magenta/input/interpolation1.mid --input_midi_2=C:/Users/USER/magenta/input/interpolation2.mid --output_dir=C:/Users/USER/magenta/output

<br/>
<br/>
<br/>


## [ custom data sequence ]

convert_dir_to_note_sequences --input_dir=C:/magenta_tutorial/input/groove_onlymidi/drummer1/session1 --output_file=C:/magenta_tutorial/custom_tfrecord/onlymidi_record.tfrecord --recursive

<br/>
<br/>
<br/>

 

## [ train ]

music_vae_train --config=groovae_4bar --run_dir=C:/magenta_tutorial/run_dir --mode=train --examples_path=C:/magenta_tutorial/custom_tfrecord/onlymidi_record.tfrecord

master = ''  
examples_path  
tfds_name  
run_dir  
num_steps  
eval_num_batches  
checkpoints_to_keep = 100  
keep_checkpoint_every_n_hours = 1  
mode = 'train'  
config = ''  
hparams = ''  
cache_dataset = True  
task = 0  
num_ps_tasks = 0  
num_sync_workers = 0  
eval_dir_suffix = ''  
log = INFO  


run_dir 경로 안에 train 폴더를 만든 다음 checkpoint를 저장하므로,  
run_dir을 적절하게 설정해줄 것


!  
checkpoint 파일은 아래 세 개가 한 쌍입니다  
model.ckpt-14.data-00000-of-00001  
model.ckpt-14.index  
model.ckpt-14.meta  
!

#
magenta에서 제공하는 checkpoint를 살펴보면, 대부분이 epoch이 3000에 가깝습니다  
허나 저는 epoch을 170에서 멈췄고 그 결과 굉장히 심심한 드럼 트랙이 출력되었습니다  
A Hierarchical Latent Structure for Variational Conversation Modeling 논문에 의하면 오버피팅을 걱정할 필요가 줄었을텐데,  
왜 epock를 다른 checkpoint와 다르게 170만 했는가라 물으신다면,  
제 개인 컴퓨터 ryzen 5950x cpu로 학습하는데 cpu를 거의 100% 사용해서 컴퓨터 수명을 위해 epoch을 낮췄습니다  
이해해주시면 감사하겠습니다  
#




[ custom generate ]

music_vae_generate --config=groovae_4bar --checkpoint_file=C:/magenta_tutorial/checkpoints/custom.tar \ --mode=sample \ --num_outputs=5 \ --output_dir=C:/magenta_tutorial/output



after:


open the online sequenser shortcut
click import midi
select predict midi file
click continue button

You can check the midi file by pressing the play button


[online_sequencer]('C:\magenta_tutorial\20220707_035419.png')