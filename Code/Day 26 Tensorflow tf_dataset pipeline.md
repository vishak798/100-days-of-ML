```python
import tensorflow as tf
```

### Create a tf dataset from list 


```python
daily_sales_numbers = [21, 22, -108, 31, -1, 32, 34,31]
tf_dataset = tf.data.Dataset.from_tensor_slices(daily_sales_numbers)
tf_dataset
```




    <TensorSliceDataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>



### Iterate through tf dataset


```python
for sales in tf_dataset:
    print(sales.numpy())
```

    21
    22
    -108
    31
    -1
    32
    34
    31
    

### Iterate through elements as numpy elements


```python
for sales in tf_dataset.as_numpy_iterator():
    print(sales)
```

    21
    22
    -108
    31
    -1
    32
    34
    31
    

### Iterate through first n elements in tf dataset


```python
for sales in tf_dataset.take(3):
    print(sales.numpy())
```

    21
    22
    -108
    

### Filter sales numbers that are < 0


```python
tf_dataset = tf_dataset.filter(lambda x: x>0)
for sale in tf_dataset.as_numpy_iterator():
    print(sale)
```

    21
    22
    31
    32
    34
    31
    

### Convert sales numbers from USA dollars to Indian Rupees (INR) Assuming 1->72 conversation rate


```python
tf_dataset = tf_dataset.map(lambda x: x*72)
for sales in tf_dataset.as_numpy_iterator():
    print(sales)
```

    1512
    1584
    2232
    2304
    2448
    2232
    

### Shuffle (randomly shuffle the dataset)


```python
tf_dataset = tf_dataset.shuffle(buffer_size=3)
for sales in tf_dataset.as_numpy_iterator():
    print(sales)
```

    1584
    1512
    2448
    2304
    2232
    2232
    

### Batching


```python
for sales_batch in tf_dataset.batch(batch_size=3):
    print(sales_batch.numpy())
```

    [1512 2232 2304]
    [2448 2232 1584]
    

### Perform all of the above operations in one shot


```python
tf_dataset = tf.data.Dataset.from_tensor_slices(daily_sales_numbers)
tf_dataset = tf_dataset.filter(lambda x: x>0).map(lambda y:y*72).shuffle(buffer_size=2).batch(batch_size=2)
for sales in tf_dataset.as_numpy_iterator():
    print(sales)
```

    [1584 2232]
    [2304 2448]
    [1512 2232]
    

## Images


```python
images_ds = tf.data.Dataset.list_files("images/*/*", shuffle=False)
```


```python
images_count = len(images_ds)
images_count
```




    130




```python
for file in images_ds.take(3):
    print(file.numpy())
```

    b'images\\cat\\20 Reasons Why Cats Make the Best Pets....jpg'
    b'images\\cat\\7 Foods Your Cat Can_t Eat.jpg'
    b'images\\cat\\A cat appears to have caught the....jpg'
    


```python
class_names = ["dog", "cat"]
```


```python
train_size = int(len(images_ds)*0.8)
train_ds = images_ds.take(train_size)
test_ds = images_ds.skip(train_size)
```


```python
for file in train_ds.as_numpy_iterator():
    print(file)
```

    b'images\\cat\\20 Reasons Why Cats Make the Best Pets....jpg'
    b'images\\cat\\7 Foods Your Cat Can_t Eat.jpg'
    b'images\\cat\\A cat appears to have caught the....jpg'
    b'images\\cat\\Adopt-A-Cat Month\xc2\xae - American Humane....jpg'
    b'images\\cat\\All About Your Cat_s Tongue.jpg'
    b'images\\cat\\Alley Cat Allies _ An Advocacy....jpg'
    b'images\\cat\\Are Cats Domesticated_ _ The New Yorker.jpg'
    b'images\\cat\\Cat Advice _ Collecting a Urine Sample....jpg'
    b'images\\cat\\Cat Throwing Up_ Normal or Cause for....jpg'
    b'images\\cat\\Cat intelligence - Wikipedia.jpg'
    b'images\\cat\\Cats Care About People More Than Food....jpg'
    b'images\\cat\\Cats _ The Humane Society of the United....jpg'
    b'images\\cat\\Cats really do need their humans_ even....jpg'
    b'images\\cat\\China_s First Cloned Kitten_ Garlic....png'
    b'images\\cat\\Famous Cat Performances in Movies_ Ranked.jpg'
    b'images\\cat\\Giving cats food with an antibody may....jpg'
    b'images\\cat\\Home_ sweet home_ How to bring an....jpg'
    b'images\\cat\\How to Determine Your Cat_s Age.jpg'
    b'images\\cat\\How to buy the best cat food_ according....jpg'
    b'images\\cat\\International Cat Care _ The ultimate....jpg'
    b'images\\cat\\Is My Cat Normal_.jpg'
    b'images\\cat\\Learn what to do with Stray and Feral....jpg'
    b'images\\cat\\New Cat Checklist 2021_ Supplies for....jpg'
    b'images\\cat\\Orlando Cat Caf\xc3\xa9.png'
    b'images\\cat\\Pet Insurance for Cats & Kittens _ Petplan.png'
    b'images\\cat\\Reality check_ Can cat poop cause....jpg'
    b'images\\cat\\Soon_ the internet will make its own....jpg'
    b'images\\cat\\Stray Cat Alliance \xc2\xbb Building a No Kill....jpg'
    b'images\\cat\\Texas lawyer accidentally uses cat....jpg'
    b'images\\cat\\The 10 Best Types of Cat _ Britannica.jpg'
    b'images\\cat\\The Cat Health Checklist_ Everything....jpg'
    b'images\\cat\\The Joys of Owning a Cat - HelpGuide.org.jpg'
    b'images\\cat\\The Science-Backed Benefits of Being a....jpg'
    b'images\\cat\\Thinking of getting a cat....png'
    b'images\\cat\\Urine Marking in Cats _ ASPCA.jpg'
    b'images\\cat\\Want your cat to stay in purrrfect....jpg'
    b'images\\cat\\What does the COVID-19 summer surge....jpg'
    b'images\\cat\\What to do if your cat is marking....jpg'
    b'images\\cat\\Why Cats Sniff Rear Ends _ VCA Animal....png'
    b'images\\cat\\Why Do Cats Hate Water_ _ Britannica.jpg'
    b'images\\dog\\10 Teacup Dog Breeds for Tiny Canine Lovers.jpg'
    b'images\\dog\\100_ Dogs Pictures _ Download Free....jpg'
    b'images\\dog\\11 Things Humans Do That Dogs Hate.jpg'
    b'images\\dog\\15 Amazing Facts About Dogs That Will....jpg'
    b'images\\dog\\20 must-have products for new dog owners.jpg'
    b'images\\dog\\25 Best Small Dog Breeds \xe2\x80\x94 Cute and....jpg'
    b'images\\dog\\25 Low-Maintenance Dog Breeds for....jpg'
    b'images\\dog\\2nd pet dog tests positive for COVID-19....jpg'
    b'images\\dog\\356 Free Dog Stock Photos - CC0 Images.jpg'
    b'images\\dog\\45 Best Large Dog Breeds - Top Big Dogs_yyth....jpg'
    b'images\\dog\\50 Cutest Dog Breeds as Puppies....jpg'
    b'images\\dog\\50 dog breeds and their history that....jpg'
    b'images\\dog\\66 gifts for dogs or dog lovers to get_yythk....jpg'
    b'images\\dog\\7 Tips on Canine Body Language _ ASPCApro.jpg'
    b'images\\dog\\8 amazing Indian dog breeds that....png'
    b'images\\dog\\9 Reasons to Own a Dog.jpg'
    b'images\\dog\\AKC Pet Insurance _ Health Insurance....png'
    b'images\\dog\\Aggression in dogs _ Animal Humane Society.jpg'
    b'images\\dog\\Ancient dog DNA reveals 11_000 years of....jpg'
    b'images\\dog\\Are Dogs Really Color-Blind_ _ Britannica.jpg'
    b'images\\dog\\Best Dog & Puppy Health Insurance Plans....jpg'
    b'images\\dog\\Best Hypoallergenic Dogs [Updated....jpg'
    b'images\\dog\\Body Condition Score....jpg'
    b'images\\dog\\Calculate Your Dog_s Age With This New....jpg'
    b'images\\dog\\Canine Mind....jpg'
    b'images\\dog\\Carolina Dog Dog Breed Information....jpg'
    b'images\\dog\\Cats and Dogs.jpg'
    b'images\\dog\\Colitis in Dogs _ VCA Animal Hospital.jpg'
    b'images\\dog\\Common Dog Breeds and Their Health Issues.jpg'
    b'images\\dog\\Dog - Role in human societies _ Britannica.jpg'
    b'images\\dog\\Dog Breed Chart....jpg'
    b'images\\dog\\Dog Breeds Banned By Home Insurance....jpg'
    b'images\\dog\\Dog collars _ The Humane Society of the....jpg'
    b'images\\dog\\Dogs _ Healthy Pets_ Healthy People _ CDC.jpg'
    b'images\\dog\\Dogs caught coronavirus from their....jpg'
    b'images\\dog\\First dog Major back at White House....jpg'
    b'images\\dog\\Genes contribute to dog breeds_ iconic....jpg'
    b'images\\dog\\Germany_ Dogs must be walked twice a....jpg'
    b'images\\dog\\Great Dane - Wikipedia.jpg'
    b'images\\dog\\Haunted Victorian Child_ Dog....jpg'
    b'images\\dog\\Hong Kong dog causes panic \xe2\x80\x93 but here_s... (1).jpg'
    b'images\\dog\\Hong Kong dog causes panic \xe2\x80\x93 but here_s....jpg'
    b'images\\dog\\Hot dogs_ what soaring puppy thefts....jpg'
    b'images\\dog\\How Many Dog Breeds Are There_ _ Hill_s Pet.jpg'
    b'images\\dog\\How My Dog Knows When I_m Sick - The....jpg'
    b'images\\dog\\How To Read Your Dog_s Body Language....png'
    b'images\\dog\\How dogs contribute to your health and....jpg'
    b'images\\dog\\How to make your dog feel comfortable....jpg'
    b'images\\dog\\Important Things Every Dog Owner Should....jpg'
    b'images\\dog\\Largest Dog Breeds \xe2\x80\x93 American Kennel Club.jpg'
    b'images\\dog\\List of Dog Breeds _ Petfinder.jpg'
    b'images\\dog\\List of dog breeds - Wikipedia.jpg'
    b'images\\dog\\Maltese Dog Breed Information_ Pictures....jpg'
    b'images\\dog\\Modern Dog magazine _ the best dog....jpg'
    b'images\\dog\\Mood-Boosting Benefits of Pets....jpg'
    b'images\\dog\\Most Expensive Dog Breeds For Pet....png'
    b'images\\dog\\Most Popular Breeds \xe2\x80\x93 American Kennel Club.jpg'
    b'images\\dog\\Most Popular Dog Breeds According....jpg'
    b'images\\dog\\Most Popular Dog Names of 2020....jpg'
    b'images\\dog\\Puppy Dog Pictures _ Download Free....jpg'
    b'images\\dog\\Rescue turns dog with untreatable tumor....jpg'
    b'images\\dog\\Rottweiler Dog Breed Information....jpg'
    b'images\\dog\\Science_ Talking to Your Dog Means You....jpg'
    b'images\\dog\\Service Dogs from Southeastern Guide Dogs.jpg'
    

### Function to get the label of the image


```python
import os
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]
```

### Function to get the label and read the image


```python
def process_image(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path) # load the raw data from the file as a string
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [128, 128])
    return img, label
```


```python
img, label = process_image("images\\cat\\20 Reasons Why Cats Make the Best Pets....jpg")
img.numpy().shape
```




    (128, 128, 3)



### Mapping the function


```python
train_ds = train_ds.map(process_image)
test_ds = test_ds.map(process_image)
```

### Scaling the image 


```python
def scale(image,label):
    return image/255, label
    
```


```python
train_ds = train_ds.map(scale)
```


```python
for image, label in train_ds.take(5):
    print("****Image: ",image.numpy()[0][0])
    print("****Label: ",label.numpy())
```

    ****Image:  [0.60784316 0.7294118  0.84313726]
    ****Label:  b'cat'
    ****Image:  [0.6878983  0.69181985 0.672212  ]
    ****Label:  b'cat'
    ****Image:  [0.7736826  0.8089767  0.79120713]
    ****Label:  b'cat'
    ****Image:  [0.38330558 0.6778809  0.3122568 ]
    ****Label:  b'cat'
    ****Image:  [0.03924632 0.07846201 0.04316789]
    ****Label:  b'cat'
    
