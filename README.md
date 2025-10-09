# KALEIDO: OPEN-SOURCED MULTI-SUBJECT REFERENCE VIDEO GENERATION MODEL
This repository is intended to store the code and ckpt for Kaleido .

<div align="center">
<img src='resources/examples_show.png'>
</div>

## Update and News
* 2025.10.9: 🔥 We propose **Kaleido**, a novel multi-subject reference video generation model.

## Qucik Start

### Prompt Optimization

Before running the model, please refer to this guide to see how we use large models like GLM-4.5 (or other comparable products, such as GPT-5) to optimize the model. This is crucial because the model is trained with long prompts, and a good prompt directly impacts the quality of the video generation.

### Diffusers

**Please make sure your Python version is between 3.10 and 3.12, inclusive of both 3.10 and 3.12.**

```
pip install -r requirements.txt
```

### Checkpoints Download

**Note:** Due to double-blind review requirements, we do not provide the checkpoint download link here. 

| ckpts       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Kaleido-1.3B      | Soon   | Supports both 480P and 512P
| Kaleido-14B | Soon | Supports both 512P and 720P

### Inference

```
python sample_video.py --base configs/video_model/dit_crossattn_14B_wanvae.yaml configs/sampling/sample_wanvae_concat_14b.yaml
```
**Note:** The condition input txt file should contain lines in the following format:
```
prompt@@image1.png@@image2.png@@image3.png
```

### Training

```
python train_video_concat.py --base configs/video_model/dit_crossattn_14B_wanvae.yaml configs/training/video_wabx_14B_concat.yaml
```
**Note:** Our training strategy is based on the CogivideoX model. For detailed information about the training process, please refer to the [CogivideoX repository](https://github.com/zai-org/CogVideo).

## Gallery
Our model is capable of broadly referencing various types of images, including humans, objects, and diverse scenarios such as try-on. This demonstrates its versatility and generalization ability across different tasks.

<table style="width: 100%; border-collapse: collapse; text-align: center; border: 1px solid #ccc;">
  <tr>
    <th style="text-align: center;">
      <strong>Reference Images</strong>
    </th>
    <th style="text-align: center;">
      <strong>Kaleido Results (512P)</strong>
    </th>
  </tr>

  <!-- <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/1/image1.jpg" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/1/101.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr> -->

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/2/image1.jpg" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/2/2.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/3/image1.jpg" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/3/image2.jpg" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/3/6.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/4/image1.jpg" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/4/image2.jpg" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/4/62.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr>


  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/5/image1.jpg" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/5/image2.jpg" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/5/109.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/6/image1.jpg" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/6/image2.jpg" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/6/120.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr>
  
  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/7/subject_0.png" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/7/subject_1.png" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/7/subject_2.png" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/7/output.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/8/image1.jpg" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/8/image2.jpg" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/8/image3.jpg" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/8/140.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/9/subject_0.png" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/9/subject_1.png" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/9/subject_2.png" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/9/output.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="resources/512p/10/subject_0.png" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/10/subject_1.png" alt="Image 1" style="height: 150px;">
      <img src="resources/512p/10/subject_2.png" alt="Image 1" style="height: 150px;">
    </td>
    <td>
      <video src='resources/512p/10/output.mp4' style="height: 150px;" controls autoplay loop></video>
    </td>
  </tr>
</table>

## Todo List
- [x] Inference codes and Training codes for Kaleido
- [ ] Datapipline of Kaleido
- [ ] Checkpoint of Kaleido
