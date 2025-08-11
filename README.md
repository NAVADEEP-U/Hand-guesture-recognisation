import argparse
        prob_text = ''

        if lm is not None:
            X = lm.reshape(1, -1)
            Xs = scaler.transform(X)
            probs = model.predict(Xs, verbose=0)[0]
            idx = np.argmax(probs)
            label = le.inverse_transform([idx])[0]
            prob = probs[idx]
            recent_preds.append(label)

            # vote
            if recent_preds:
                votes = {}
                for p in recent_preds:
                    votes[p] = votes.get(p, 0) + 1
                voted_label = max(votes.items(), key=lambda x: x[1])[0]
                label_text = voted_label
                prob_text = f'{prob*100:.1f}%'

            # draw landmarks
            for handlms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, handlms, mp_hands.HAND_CONNECTIONS)
        else:
            recent_preds.clear()

        cv2.putText(image, f'Pred: {label_text} {prob_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('Gesture Inference - press q to quit', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()




def main():
    parser = argparse.ArgumentParser(description='Gesture recognizer for simple sign words')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    p_collect = subparsers.add_parser('collect')
    p_collect.add_argument('--label', required=True, help='Label name for this gesture (e.g., water)')
    p_collect.add_argument('--count', type=int, default=200, help='Number of frames/samples to collect')
    p_collect.add_argument('--wait', type=int, default=2, help='Seconds to wait before starting')

    p_train = subparsers.add_parser('train')
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch', type=int, default=32)

    p_infer = subparsers.add_parser('infer')

    args = parser.parse_args()

    if args.mode == 'collect':
        collect(args.label, args.count, args.wait)
    elif args.mode == 'train':
        train(epochs=args.epochs, batch_size=args.batch)
    elif args.mode == 'infer':
        infer()


if __name__ == '__main__':
    main()
