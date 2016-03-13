
# This is the server logic for a Shiny web application.
# You can find out more about building applications with Shiny here:
#
# http://shiny.rstudio.com
#


shinyServer(function(input, output) {


python.exec("
#Random initial angles
angle1 = random()
angle2 = random()

def get_pair():
  global sliding_window
  sliding_window.append(get_sample())
  input_value = sliding_window[0]
  output_value = sliding_window[-1]
  sliding_window = sliding_window[1:]
  return input_value, output_value
input_dim = 2
            
#To maintain state
last_value = array([0 for i in range(input_dim)])
last_derivative = array([0 for i in range(input_dim)])

def get_total_input_output():
  global last_value, last_derivative
  raw_i, raw_o = get_pair()
  raw_i = raw_i[0]
  l1 = list(raw_i)
  derivative = raw_i - last_value
  l2 = list(derivative)
  last_value = raw_i
  l3 = list(derivative - last_derivative)
  last_derivative = derivative
  return array([l1 + l2 + l3]), raw_o

#TF Setup
#Input
input_layer = tf.placeholder(tf.float32, [1, input_dim*3])

#The Weights and Biases matrices first
output_W1 = tf.Variable(tf.truncated_normal([input_dim*3, input_dim]))
output_b1 = tf.Variable(tf.zeros([input_dim]))
") 


observeEvent(input$reset, {
iw <- as.numeric(input$input_wave)-1
python.assign( "input_wave", iw )
python.assign( "prlag", input$prlag )
python.exec("
lag = prlag            
def get_sample():
  global angle1, angle2
  angle1 += 2*pi/float(frequency1)
  angle2 += 2*pi/float(frequency2)
  angle1 %= 2*pi
  angle2 %= 2*pi
  return array([array([
  5 + 5*sin(angle1) + 10*cos(angle2)**input_wave,
  7 + 7*sin(angle2)**input_wave + 14*cos(angle1)])])

sliding_window = []
for i in range(lag - 1):
  sliding_window.append(get_sample())
")

ct <- as.numeric(input$cell_type)
python.assign('cell_type', ct)

python.exec("
#The Cell initialization
if cell_type == 1:
  lstm_layer1 = rnn_cell.BasicRNNCell(input_dim*3)
  lstm_state1 = tf.Variable(tf.zeros([1, lstm_layer1.state_size]))
  lstm_output1, lstm_state_output1 = lstm_layer1(input_layer, lstm_state1, scope='BasicRNN')
  lstm_update_op1 = lstm_state1.assign(lstm_state_output1)
  final_output = tf.matmul(lstm_output1, output_W1) + output_b1
if cell_type == 2:
  lstm_layer2 = rnn_cell.GRUCell(input_dim*3)
  lstm_state2 = tf.Variable(tf.zeros([1, lstm_layer2.state_size]))
  lstm_output2, lstm_state_output2 = lstm_layer2(input_layer, lstm_state2, scope='GRUCell')
  lstm_update_op2 = lstm_state2.assign(lstm_state_output2)
  final_output = tf.matmul(lstm_output2, output_W1) + output_b1
if cell_type == 3:
  lstm_layer3 = rnn_cell.BasicLSTMCell(input_dim*3)
  lstm_state3 = tf.Variable(tf.zeros([1, lstm_layer3.state_size]))
  lstm_output3, lstm_state_output3 = lstm_layer3(input_layer, lstm_state3, scope='BasicLSTM')
  lstm_update_op3 = lstm_state3.assign(lstm_state_output3)
  final_output = tf.matmul(lstm_output3, output_W1) + output_b1
if cell_type == 4:
  lstm_layer4 = rnn_cell.BasicLSTMCell(input_dim*3)
  lstm_layer4 = rnn_cell.MultiRNNCell([lstm_layer4]*2)
  lstm_layer4 = rnn_cell.DropoutWrapper(lstm_layer4,output_keep_prob=0.8)
  lstm_state4 = tf.Variable(tf.zeros([1, lstm_layer4.state_size]))
  lstm_output4, lstm_state_output4 = lstm_layer4(input_layer, lstm_state4, scope='LSTM-2')
  lstm_update_op4 = lstm_state4.assign(lstm_state_output4)
  final_output = tf.matmul(lstm_output4, output_W1) + output_b1

##Input for correct output (for training)
correct_output = tf.placeholder(tf.float32, [1, input_dim])

##Calculate the Sum-of-Squares Error
error = tf.pow(tf.sub(final_output, correct_output), 2)

##The Optimizer
#Adam works best
train_step = tf.train.AdamOptimizer(0.0006).minimize(error)

w_hist = tf.histogram_summary('weights', output_W1)
b_hist = tf.histogram_summary('biases', output_b1)
y_hist = tf.histogram_summary('y', final_output)
")

python.exec("
##Session
sess = tf.Session()
merged = tf.merge_summary([w_hist,b_hist,y_hist])
tf.train.write_graph(sess.graph_def,'/tmp/tensor','graph.txt')
writer = tf.train.SummaryWriter('/tmp/tensor/',sess.graph_def)
init = tf.initialize_all_variables()
sess.run(init)
##Training
actual_output1 = []
actual_output2 = []
network_output1 = []
network_output2 = []
x_axis = []
")

})

df_results <- eventReactive(input$train, {
python.assign( "iter", input$iter)
python.exec("
i_end = iter + len(x_axis)
for i in range(i_end):
  input_v, output_v = get_total_input_output()
  if cell_type == 1:
    _, _, network_output, merge = sess.run([lstm_update_op1,
                                          train_step,
                                          final_output,merged],
                                         feed_dict = {
                                           input_layer: input_v,
                                           correct_output: output_v})
  if cell_type == 2:
    _, _, network_output, merge = sess.run([lstm_update_op2,
                                          train_step,
                                          final_output,merged],
                                         feed_dict = {
                                           input_layer: input_v,
                                           correct_output: output_v})
  if cell_type == 3:
      _, _, network_output, merge = sess.run([lstm_update_op3,
                                          train_step,
                                          final_output,merged],
                                         feed_dict = {
                                           input_layer: input_v,
                                           correct_output: output_v})
  if cell_type == 4:
      _, _, network_output, merge = sess.run([lstm_update_op4,
                                          train_step,
                                          final_output,merged],
                                         feed_dict = {
                                           input_layer: input_v,
                                           correct_output: output_v})
  actual_output1.append(output_v[0][0])
  actual_output2.append(output_v[0][1])
  network_output1.append(network_output[0][0])
  network_output2.append(network_output[0][1])
  x_axis.append(i)
  writer.add_summary(merge, i)
out2 = np.array(network_output2).tolist()
out1 = np.array(network_output1).tolist()
")
x_axis <- python.get("x_axis")
actual_output1 <- python.get("actual_output1")
actual_output2 <- python.get("actual_output2")
pred_output1 <- python.get("out1")
pred_output2 <- python.get("out2") 
df<- as.data.frame(cbind(x_axis,actual_output1,pred_output1,actual_output2,pred_output2))
df
})

output$out1 <- renderPlotly({
  df <- df_results()
  mse1 <- Metrics::mse(df$pred_output1, df$actual_output1)
  title1 <- paste("Series X, MSE:",mse1,sep = " ")
p<- plot_ly(x = x_axis, y = actual_output1, data = df, name = "Actual") %>%
    add_trace(x = x_axis, y = pred_output1, data = df, name = "Predicted") %>%
    layout(title = title1)
p
})

output$out2 <- renderPlotly({
  df <- df_results()
  mse2 <- mse(df$pred_output2, df$actual_output2)
  title2 <- paste("Series Y, MSE:",mse2,sep = " ")
  p<- plot_ly(x = x_axis, y = actual_output2, data = df,name = "Actual") %>%
    add_trace(x = x_axis, y = pred_output2, data = df,name = "Predicted") %>%
    layout(title = title2)
  p
})

output$df <- DT::renderDataTable(
  DT::datatable(df_results())
)
  
})
