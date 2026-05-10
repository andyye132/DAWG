## **summary**

We're training a binary classifier that flags rendered webpages whose pixels have been adversarially modified to fool MolmoWeb, a screenshot-only visual web agent. To train it, we generate a large dataset of (clean, attacked) screenshot pairs by 
- (a) rendering synthetic webpages that mimic real-world categories 
- (b) using white-box PGD against MolmoWeb-4B to produce an adversarial PNG patch per example. (PGD: https://medium.com/@zachariaharungeorge/unveiling-the-power-of-projected-gradient-descent-in-adversarial-attacks-2f92509dde3c, https://www.stat.cmu.edu/~siva/teaching/725/lec6.pdf, )
- (c) re-rendering the page with that patch injected as an HTML `<img>` overlay. 

The trained classifier wraps MolmoWeb at inference: if a page is flagged as adversarial, the agent refuses to act and outputs something like "this webpage seems fishy!!"

---
# **initial spec**

**Our threat model** will be control over the server side rendering of a webpage. This server side code will contain a adversarial image overlay, which contains PGD-optimized pixels alongside the legitimate content. This accounts for bad actors who own the websites; they're able to seed their sites with masks which screw with web agents.

**Our sites** will be synthetic templates which mimic real world websites. We plan to have a diverse set of sites such as news, shop, blog, profile, dashboard, login, and search results. We will build these websites, and keep them at size 1280x720 (MW takes screenshots at these dimensions). We will generate these sites in bulk with Claude.

# **masking**

**Our mask** will be injected into the page HTML at render time. The position and size of this image will be randomized per example, and we can work to make this element aware (try to hide in certain parts of the screen). The mask content will be a PNG whose pixels are optimized via PGD against MW's vision (it is an open source model). There are different options for masks:
1. **Pixel Replacement:** Simply place a small patch of pixels which throw it off. IMO this is the weakest since human inspection throws it off.
2. **Patch Pertubation:** take the original pixels in that region and add a small pertubation to each pixel (like the panda!). This would be hard to do with text. We can constrain it to non-text locations. Which brings us to...
3. **Smart positioning:** We have to figure out how to place these masks in disguised places. A picture of a busy street would be better than against a white background. 

### **Our attack levels fall under three categories:**
1. **L1 returns bogus**. We simply wish to maximize cross entropy against MW's clean output tokens (AKA, get our output as far as possible from the correct output). This would defend against bad actors which want websites to output bogus.
2. **L2 returns plausibly correct items**. We want the output to provide plausibly correct misinformation (this tomato costs 900$). This is dangerous because the bad actor could skew the information the website returns to chatbot queries (for example, pushing towards advertisement; 'you can get this for cheaper at scamsite.com!'), and the user might fall for it.
3. **L3 gets MW to perform actions.** MolmoWeb operates a website like a human would; and thus is able to scroll, click buttons, and more. The goal of this level of attack is to get MW to click a button on the website that, maybe, does something bad. Or not. The main point to prove is that if we can get a query agent to consistently click a button, then no matter what this button does, this is a serious problem in webquery security.

We focus on L1 for now. If we can get the discriminator to defend against L1, then L2 and L3 should not be hard. We can just show that it is robust by defending against all 3, and if MW fails on all L1,L2,L3 then this is a non-trivial improvement.
### **Our discriminator will be built upon DINOv2**

DINOv2 is a ViT: https://arxiv.org/abs/2304.07193 
We can also just do a 2 layer MLP head which is a binary classifier based on 1280x720 screenshots. Cap or no cap.

# **data**
Our data looks like: (clean_screenshot, masked_screenshot, query, clean_output, attacked_output, patch_bbox, target). Ok why each?
- **(clean_screenshot, masked_screenshot)**: this is what the discriminator sees. It's goal is to find quirks and problems within the webpage, so the discriminator needs to learn differences between mask and no mask; this is the whole point
- **(query, clean_output, attacked_output)**: this is used to filter out our clean screenshot vs masked screenshot pairs. If clean output == attacked output then the pair of screenshots is useless.
- **(patch_bbox, target)**: this is free for us to collect and can be useful to learn to find where patches are, and target would be what PGD is optimized towards (maybe we want to make a tomato cost 90000$)

### **how can we check clean != attacked?**
LLM's have near infinite degrees of freedom. Similar answers could have the same meaning but be typed differently. How can we check that answers with same meaning are marked as 'the same'? 
We place them all in latent space and have some radius of L1 distance between embeddings and if its within this radius then its right. Formally we have sentence embeddings and check via cosine similarity.
# **cool beyond the MVP:**
1. server side agent fingerprinting... there is tech to render the mask ONLY when the web agent visits the site! the human would see only the clean site and the mask is only for agent. This is a full real world threat and would be really cool if we could have a demo with this and if you click in
2. real site demo... we can build a demo where we put two links (benign and tampered) and have a side by side molmoweb interacting wiht them and then show MW return bogus and MW+ return 'this site can't be trusted
3. L2 and L3 if we don't get there can be future work. Proving L1 works kinda shows those two can happen too.
4. Intelligent patch placement. Right now, I think the best way to do it is to randomly place patches around the 1980x720 screen and ensure that each 1980x720 contains some patch, that way the agent is always affected. If there are optimal places to place patches, would be a great place to look.
	1. maybe it's more intelligent to have some areas with patches and some areas without; so it still seems reasonable.