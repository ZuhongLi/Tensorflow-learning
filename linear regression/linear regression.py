data_file = './datasets/birth_life_2010.txt'

# step1:read datasets
data,n_samples = utils.read_birth_life_data(data_file)

# step2:create placeholder for X(birth_rate) and Y(life_expectancy)
X = tf.placeholder(tf.float32,name='X')
Y = tf.placeholder(tf.float32,name='Y')

# step3:create weight and bias ,initialized to 0
w = tf.get_variable('Weight',initializer=tf.constant(0.0))
b = tf.get_variable('Bias',initializer=tf.constant(0.0))

# step4:construct model for prediction
predict_y = w*X+b

# step5:use the square error to be loss function
loss = tf.square(Y-predict_y,name='loss')

# step6:use gradient decent with learning rate 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    # step7:initialize global variables
    sess.run(tf.global_variables_initializer())

    #step8:train model
    for i in range(100):
        for x,y in data:
            sess.run(optimizer,feed_dict = {X:x,Y:y})

    #step9: output w and b
    out_w,out_b = sess.run([w,b])

# step10: draw the plot
plt.plot(data[:,0],data[:,1],'bo',label='Real data')
plt.plot(data[:,0],data[:,0]*out_w+out_b,'r',label = 'Predict data')
plt.legend()
plt.show()
