
# This is the user-interface definition of a Shiny web application.
# You can find out more about building applications with Shiny here:
#
# http://shiny.rstudio.com
#

shinyUI(fluidPage(

  # Application title
  titlePanel("RNN using Tensorflow"),
  sidebarLayout(
    sidebarPanel(
      selectInput("input_wave", "Input:",
                  c("Sine" = 1,
                    "Sine+Cosine" = 2,
                    "Sine+Cosine**2" = 3),selected = 2),
      numericInput("prlag", label = "Steps for prediction window:", value = 20),
      selectInput("cell_type", "Cell Type:",
                  c("Basic RNN Cell" = 1,
                    "GRU Cell" = 2,
                    "Basic LSTM Cell" = 3,
                    "2 layer LSTM" = 4),selected = 3),
      helpText("After selecting model options, use initialize button:"),
      actionButton("reset", "Initialize Model"),
      helpText("Set the # of iterations. Pressing train button again adds more interations for the same model. Takes about 2 seconds for 1000 iterations."),
      numericInput("iter", label = "Iterations:", value = 5000,max = 10000),
      actionButton("train", "Train"),
      p(),
      h4("  ",a(" About & Code",href="https://github.com/rajshah4/tensorflow_shiny"))
    ),

    # Show a plot of the generated distribution
    mainPanel(
      p(" ",a("Tensorboard Visualization",href="http://projects.rajivshah.com:6006")),
      plotlyOutput('out1'),
      plotlyOutput('out2'),
      DT::dataTableOutput('df'))
  )
))
