from objective import mask_nll_loss
from vocabulary import batch_to_training_data
from plotting import generate_plot

import torch
import torch.nn as nn
import random
import logging
import os


def train(training_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, config):
    # Extract fields from batch
    input_variable, lengths, target_variable, mask, max_target_len = training_batch

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(config["device"])
    lengths = lengths.to(config["device"])
    target_variable = target_variable.to(config["device"])
    mask = mask.to(config["device"])

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[config["SOS_token"] for _ in range(config["batch_size"])]])
    decoder_input = decoder_input.to(config["device"])

    # Set initial decoder hidden state to the encode's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = random.random() < config["teacher_forcing_ratio"]

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)

            # Calculate and accumulate loss
            mask_loss, n_total = mask_nll_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(config["batch_size"])]])
            decoder_input = decoder_input.to(config["device"])

            # Calculate and accumulate loss
            mask_loss, n_total = mask_nll_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(encoder.parameters(), config["clip"])
    nn.utils.clip_grad_norm_(decoder.parameters(), config["clip"])

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def train_iterations(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, checkpoint, config):
    # Load batches fro each iteration
    training_batches = [batch_to_training_data(voc, [random.choice(pairs) for _ in range(config["batch_size"])])
                        for _ in range(config["n_iteration"])]

    logging.info("Initializing")
    start_iteration = 1
    print_loss = 0
    graph_losses = []
    if checkpoint:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    logging.info("Training...")
    for iteration in range(start_iteration, config["n_iteration"] + 1):
        training_batch = training_batches[iteration - 1]

        # Run a training iteration with batch
        loss = train(training_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, config)
        print_loss += loss
        graph_losses.append(loss)

        # Print progress
        if iteration % config["print_every"] == 0:
            print_loss_avg = print_loss / config["print_every"]
            logging.info("Iteration: {}; percent complete {:.1f}%; Average loss: {:.4f}".format(
                iteration, iteration / config["n_iteration"] * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if iteration % config["save_every"] == 0:
            logging.info("Saving model to file...")
            if not os.path.exists(config["directory"]):
                os.makedirs(config["directory"])
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': decoder.embedding.state_dict()
            }, os.path.join(config["directory"], '{}_{}.tar'.format(iteration, 'checkpoint')))

    # Plot loss over epochs
    logging.info("Done. Plotting loss. Say Hi...")
    generate_plot(graph_losses)
